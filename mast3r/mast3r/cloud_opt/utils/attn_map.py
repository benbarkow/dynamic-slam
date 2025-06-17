import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2
import os
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.filters import threshold_multiotsu
from dust3r.cloud_opt.commons import (get_imshapes, NoGradParamDict, edge_str)

from skimage.filters import threshold_otsu, threshold_multiotsu




class AttentionMaskGenerator:

    def __init__(
            self,
            res,
            view1,
            view2
    ):
        self.pred1 = res[0]
        self.pred2 = res[1]
        print(view1)

        """
        if isinstance(view1['idx'], int):
            view1['idx'] = [view1['idx']]
        elif hasattr(view1['idx'], 'tolist'):
            view1['idx'] = view1['idx'].tolist()

        if isinstance(view2['idx'], int):
            view2['idx'] = [view2['idx']]
        elif hasattr(view2['idx'], 'tolist'):
            view2['idx'] = view2['idx'].tolist()
        self.edges = [(int(i), int(j)) for i, j in zip(view1['idx'], view2['idx'])]
        self.is_symmetrized = set(self.edges) == {(j, i) for i, j in self.edges}
        """
        self.edges = [(0, 1), (1, 0)]
        #print("edges:", self.edges)
        pred1_pts = self.pred1['pts3d']
        pred2_pts = self.pred2['pts3d_in_other_view']
        
        self.pred_i = NoGradParamDict({k: pred1_pts[n] for n, k in enumerate(self.str_edges)})
        self.pred_j = NoGradParamDict({k: pred2_pts[n] for n, k in enumerate(self.str_edges)})
        self.imshapes = get_imshapes(self.edges, pred1_pts, pred2_pts)
        #print(self.imshapes)
        pass
 
    def set_cross_att(self):
        pred1, pred2 = self.pred1, self.pred2
        cross_att_k_i_mean, cross_att_k_i_var, cross_att_k_j_mean, cross_att_k_j_var = self.aggregate_attention_maps(pred1, pred2)

        def fuse_attention_channels(att_maps):
            # att_maps: B, H, W, C
            # normalize
            att_maps_min = att_maps.min()
            att_maps_max = att_maps.max()
            att_maps_normalized = (att_maps - att_maps_min) / (att_maps_max - att_maps_min + 1e-6)
            # average channel
            att_maps_fused = att_maps_normalized.mean(dim=-1) # B, H, W
            # normalize
            att_maps_fused_min = att_maps_fused.min()
            att_maps_fused_max = att_maps_fused.max()
            att_maps_fused = (att_maps_fused - att_maps_fused_min) / (att_maps_fused_max - att_maps_fused_min + 1e-6)
            return att_maps_normalized, att_maps_fused

        self.cross_att_k_i_mean, self.cross_att_k_i_mean_fused = fuse_attention_channels(cross_att_k_i_mean)
        self.cross_att_k_i_var, self.cross_att_k_i_var_fused = fuse_attention_channels(cross_att_k_i_var)
        self.cross_att_k_j_mean, self.cross_att_k_j_mean_fused = fuse_attention_channels(cross_att_k_j_mean)
        self.cross_att_k_j_var, self.cross_att_k_j_var_fused = fuse_attention_channels(cross_att_k_j_var)

    def aggregate_attention_maps(self, pred1, pred2):
        #print(pred1)

        def aggregate_attention(attention_maps, aggregate_j=True):
            attention_maps = NoGradParamDict({ij: nn.Parameter(attention_maps[n], requires_grad=False)
                                            for n, ij in enumerate(self.str_edges)})
            #print("attention maps: ")
            #print(attention_maps)
            #print(len(attention_maps))
            aggregated_maps = {}
            for edge, attention_map in attention_maps.items():
                idx = edge.split('_')[1 if aggregate_j else 0]
                att = attention_map.clone()
                if idx not in aggregated_maps:
                    aggregated_maps[idx] = [att]
                else:
                    aggregated_maps[idx].append(att)
            stacked_att_mean = [None] * len(self.imshapes)
            stacked_att_var = [None] * len(self.imshapes)
            for i, aggregated_map in aggregated_maps.items():
                att = torch.stack(aggregated_map, dim=-1)
                att[0,0] = (att[0,1] + att[1,0])/2
                stacked_att_mean[int(i)] = att.mean(dim=-1)
                stacked_att_var[int(i)] = att.std(dim=-1)
            #print("aggregated_maps keys:", aggregated_maps.keys())
            #print("length stacked_att_mean:", len(stacked_att_mean))
            #print("stacked_att_mean contents:", stacked_att_mean)
            return torch.stack(stacked_att_mean).float().detach(), torch.stack(stacked_att_var).float().detach()

        cross_att_k_i_mean, cross_att_k_i_var = aggregate_attention(pred1['cross_atten_maps_k'], aggregate_j=True)
        cross_att_k_j_mean, cross_att_k_j_var = aggregate_attention(pred2['cross_atten_maps_k'], aggregate_j=False)
        return cross_att_k_i_mean, cross_att_k_i_var, cross_att_k_j_mean, cross_att_k_j_var
    
    
    def save_attention_maps(self, save_folder='demo_tmp/attention_vis'):
        self.vis_attention_masks(1-self.cross_att_k_i_mean_fused, save_folder=save_folder, save_name='cross_att_k_i_mean')
        self.vis_attention_masks(self.cross_att_k_i_var_fused, save_folder=save_folder, save_name='cross_att_k_i_var')
        self.vis_attention_masks(1-self.cross_att_k_j_mean_fused, save_folder=save_folder, save_name='cross_att_k_j_mean')
        self.vis_attention_masks(self.cross_att_k_j_var_fused, save_folder=save_folder, save_name='cross_att_k_j_var')
        self.vis_attention_masks(self.dynamic_map, save_folder=save_folder, save_name='dynamic_map')
        self.vis_attention_masks(self.refined_dynamic_map, save_folder=save_folder, save_name='refined_dynamic_map')
        self.vis_attention_masks(self.refined_dynamic_map, save_folder=save_folder, save_name='refined_dynamic_map_labels', \
                            cluster_labels=self.dynamic_map_labels)
        
    @torch.no_grad()
    def vis_attention_masks(self, attns_fused, save_folder='demo_tmp/attention_vis', save_name='attention_channels_all_frames', cluster_labels=None):
        B, H, W = attns_fused.shape
        target_size = getattr(self, 'imshape', (H, W))

        upsampled_attns = torch.nn.functional.interpolate(attns_fused.unsqueeze(1), size=target_size, mode='nearest').squeeze(1)

        if cluster_labels is not None:
            upsampled_labels = torch.nn.functional.interpolate(cluster_labels.float().unsqueeze(1), size=target_size, mode='nearest').squeeze(1).long()

        cmap = plt.cm.get_cmap('Spectral_r')
        H_up, W_up = upsampled_attns.shape[1:]
        stacked_att_img = torch.zeros((B, 3, H_up, W_up), device=upsampled_attns.device)

        for i in range(B):
            att_np = upsampled_attns[i].cpu().numpy()
            colored_att = cmap(att_np)[:, :, :3]
            colored_att_torch = torch.from_numpy(colored_att).float().permute(2, 0, 1).to(upsampled_attns.device)
            stacked_att_img[i] = colored_att_torch

        stacked_mask = (upsampled_attns > adaptive_multiotsu_variance(upsampled_attns.cpu().numpy()))

        if cluster_labels is not None:
            num_clusters = upsampled_labels.max().item() + 1
            colors = plt.cm.tab20(np.linspace(0, 1, num_clusters))[:, :3]
            colors = torch.from_numpy(colors).float().to(upsampled_labels.device)

            stacked_mask_img = torch.zeros((B, 3, H_up, W_up), device=upsampled_labels.device)
            for i in range(num_clusters):
                mask = (upsampled_labels == i) & stacked_mask
                mask = mask.unsqueeze(1)
                stacked_mask_img += mask * colors[i].view(1, 3, 1, 1)
        else:
            stacked_mask_img = stacked_mask.unsqueeze(1).expand(-1, 3, -1, -1)

        grid_size = int(math.ceil(math.sqrt(B)))
        grid_att = torchvision.utils.make_grid(stacked_att_img, nrow=grid_size, padding=2, normalize=False)
        grid_cluster = torchvision.utils.make_grid(stacked_mask_img, nrow=grid_size, padding=2, normalize=False)
        final_grid = torch.cat([grid_att, grid_cluster], dim=1)
        os.makedirs(save_folder, exist_ok=True)
        torchvision.utils.save_image(final_grid, os.path.join(save_folder, f'0_{save_name}_fused.png'))

        fused_save_folder = os.path.join(save_folder, f'0_{save_name}_fused')
        os.makedirs(fused_save_folder, exist_ok=True)

        if B > 0:
            frames_att_dir = os.path.join(fused_save_folder, 'frames_att')
            os.makedirs(frames_att_dir, exist_ok=True)
            for i in range(B):
                att_frame = stacked_att_img[i].cpu().numpy().transpose(1, 2, 0)
                frame = (att_frame * 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{frames_att_dir}/frame_{i:04d}.png', frame)

            video_att_path = os.path.join(fused_save_folder, f'0_{save_name}_att_video.mp4')
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{frames_att_dir}/frame_%04d.png" '
                    f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                    f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                    f'-movflags +faststart -b:v 5000k "{video_att_path}"')

            frames_mask_dir = os.path.join(fused_save_folder, 'frames_mask')
            os.makedirs(frames_mask_dir, exist_ok=True)
            for i in range(B):
                mask_frame = stacked_mask_img[i].cpu().numpy().transpose(1, 2, 0)
                frame = (mask_frame * 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{frames_mask_dir}/frame_{i:04d}.png', frame)

            video_mask_path = os.path.join(fused_save_folder, f'0_{save_name}_mask_video.mp4')
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{frames_mask_dir}/frame_%04d.png" '
                    f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                    f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                    f'-movflags +faststart -b:v 5000k "{video_mask_path}"')

    
    
    @property
    def str_edges(self):
        return [edge_str(i, j) for i, j in self.edges]