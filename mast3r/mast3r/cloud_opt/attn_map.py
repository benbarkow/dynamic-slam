import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
import cv2
import os
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.filters import threshold_multiotsu
from dust3r.cloud_opt.commons import (
    get_imshapes,
    NoGradParamDict,
    edge_str,
)


class AttentionMaskGenerator:
    def __init__(self, res, view1, view2):
        self.pred1 = res[0]
        self.pred2 = res[1]
        self.pred3 = res[2]
        self.pred4 = res[3]

        print("cross shape 1",self.pred1["cross_atten_maps_k"].shape)
        print("cross shape 2",self.pred2["cross_atten_maps_k"].shape)
        self.pred1["cross_atten_maps_k"] = torch.stack((self.pred1["cross_atten_maps_k"].squeeze(0), self.pred3["cross_atten_maps_k"].squeeze(0)))
        self.pred2["cross_atten_maps_k"] = torch.stack((self.pred2["cross_atten_maps_k"].squeeze(0), self.pred4["cross_atten_maps_k"].squeeze(0)))
        print("cross shape 1",self.pred1["cross_atten_maps_k"].shape)
        print("cross shape 2",self.pred2["cross_atten_maps_k"].shape)

        view1["idx"] = [view1["idx"]]
        view2["idx"] = [view2["idx"]]
        self.edges = [
            (int(i), int(j)) for i, j in zip(view1["idx"], view2["idx"])
        ]
        print("edges", self.edges)
        self.edges = [(0, 1), (1, 0)]

        pred1_pts = self.pred1["pts3d"]
        pred3_pts = self.pred3["pts3d"]
        pred1_pts = pred1_pts.squeeze(0)
        pred3_pts = pred3_pts.squeeze(0)
        pred13_pts = torch.stack((pred1_pts, pred3_pts)) 

        pred2_pts = self.pred2["pts3d_in_other_view"]
        pred24_pts = torch.stack((pred2_pts.squeeze(0), self.pred4["pts3d_in_other_view"].squeeze(0)))
        self.pred_i = NoGradParamDict(
            {ij: pred13_pts[n] for n, ij in enumerate(self.str_edges)}
        )
        self.pred_j = NoGradParamDict(
            {ij: pred24_pts[n] for n, ij in enumerate(self.str_edges)}
        )
        self.imshapes = get_imshapes(self.edges, pred13_pts, pred24_pts)
        print(self.imshapes)
        pass

    @property
    def str_edges(self):
        return [edge_str(i, j) for i, j in self.edges]

    def set_cross_att(self):
        pred1, pred2 = self.pred1, self.pred2
        (
            cross_att_k_i_mean,
            cross_att_k_i_var,
            cross_att_k_j_mean,
            cross_att_k_j_var,
        ) = self.aggregate_attention_maps(pred1, pred2)

        def fuse_attention_channels(att_maps):
            # att_maps: B, H, W, C
            # normalize
            att_maps_min = att_maps.min()
            att_maps_max = att_maps.max()
            print("att_maps min/max:", att_maps_min, att_maps_max)
            print("att_maps contains nan:", torch.isnan(att_maps).any())
            att_maps_normalized = (att_maps - att_maps_min) / (
                att_maps_max - att_maps_min + 1e-6
            )
            # average channel
            att_maps_fused = att_maps_normalized.mean(dim=-1)  # B, H, W
            # normalize
            att_maps_fused_min = att_maps_fused.min()
            att_maps_fused_max = att_maps_fused.max()
            print("att_maps_fused min/max:", att_maps_fused_min, att_maps_fused_max)
            print("att_maps_fused contains nan:", torch.isnan(att_maps_fused).any())
            att_maps_fused = (
                att_maps_fused - att_maps_fused_min
            ) / (att_maps_fused_max - att_maps_fused_min + 1e-6)
            return att_maps_normalized, att_maps_fused

        (
            self.cross_att_k_i_mean,
            self.cross_att_k_i_mean_fused,
        ) = fuse_attention_channels(cross_att_k_i_mean)
        (
            self.cross_att_k_i_var,
            self.cross_att_k_i_var_fused,
        ) = fuse_attention_channels(cross_att_k_i_var)
        (
            self.cross_att_k_j_mean,
            self.cross_att_k_j_mean_fused,
        ) = fuse_attention_channels(cross_att_k_j_mean)
        (
            self.cross_att_k_j_var,
            self.cross_att_k_j_var_fused,
        ) = fuse_attention_channels(cross_att_k_j_var)

    def aggregate_attention_maps(self, pred1, pred2):
        def aggregate_attention(attention_maps, aggregate_j=True):
            attention_maps = NoGradParamDict({ij: nn.Parameter(attention_maps[n], requires_grad=False) 
                                            for n, ij in enumerate(self.str_edges)})
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
            return torch.stack(stacked_att_mean).float().detach(), torch.stack(stacked_att_var).float().detach()
        
        cross_att_k_i_mean, cross_att_k_i_var = aggregate_attention(pred1['cross_atten_maps_k'], aggregate_j=True)
        cross_att_k_j_mean, cross_att_k_j_var = aggregate_attention(pred2['cross_atten_maps_k'], aggregate_j=False)
        return cross_att_k_i_mean, cross_att_k_i_var, cross_att_k_j_mean, cross_att_k_j_var

    def save_attention_maps(self, save_folder='demo_tmp/attention_vis'):
        self.vis_attention_masks(1-self.cross_att_k_i_mean_fused, save_folder=save_folder, save_name='cross_att_k_i_mean')
        self.vis_attention_masks(self.cross_att_k_i_var_fused, save_folder=save_folder, save_name='cross_att_k_i_var')
        self.vis_attention_masks(1-self.cross_att_k_j_mean_fused, save_folder=save_folder, save_name='cross_att_k_j_mean')
        self.vis_attention_masks(self.cross_att_k_j_var_fused, save_folder=save_folder, save_name='cross_att_k_j_var')
        # self.vis_attention_masks(self.dynamic_map, save_folder=save_folder, save_name='dynamic_map')
        # self.vis_attention_masks(self.refined_dynamic_map, save_folder=save_folder, save_name='refined_dynamic_map')
        # self.vis_attention_masks(self.refined_dynamic_map, save_folder=save_folder, save_name='refined_dynamic_map_labels', \
        #                     cluster_labels=self.dynamic_map_labels)

    @torch.no_grad()
    def vis_attention_masks(self, attns_fused, save_folder='demo_tmp/attention_vis', save_name='attention_channels_all_frames', cluster_labels=None):

        def adaptive_multiotsu_variance(img, verbose=False):
            print(img)
            """adaptive multi-threshold Otsu algorithm based on inter-class variance maximization
            
            Args:
                img: input image array
                verbose: whether to print detailed information
                
            Returns:
                tuple: (best threshold, best number of classes)
            """
            max_classes = 4
            best_score = -float('inf')
            best_threshold = None
            best_n_classes = None
            scores = {}
            
            for n_classes in range(2, max_classes + 1):
                thresholds = threshold_multiotsu(img, classes=n_classes)
                
                regions = np.digitize(img, bins=thresholds)
                var_between = np.var([img[regions == i].mean() for i in range(n_classes)])
                
                score = var_between / np.sqrt(n_classes)
                scores[n_classes] = score
                
                if score > best_score:
                    best_score = score
                    best_threshold = thresholds[-1]
                    best_n_classes = n_classes
            
            if verbose:
                print("number of classes score:")
                for n_classes, score in scores.items():
                    print(f"number of classes {n_classes}: score {score:.4f}" + 
                        (" (best)" if n_classes == best_n_classes else ""))
                print(f"final selected number of classes: {best_n_classes}")
            
            return best_threshold

        B, H, W = attns_fused.shape
        # ensure self.imshape exists, otherwise use the original size
        target_size = getattr(self, 'imshape', (384, 512))
        print(target_size)
        
        # upsample the attention maps
        print("upsampled before", attns_fused.shape)
        print("nones", torch.isnan(attns_fused).any())
        upsampled_attns = torch.nn.functional.interpolate(
            attns_fused.unsqueeze(1),  # [B, 1, H, W]
            size=target_size, 
            mode='nearest'
        ).squeeze(1)  # [B, H', W']
        
        print("upsampled after", upsampled_attns.shape)
        print("upsampled after", upsampled_attns)
        # if there is cluster_labels, also upsample it
        if cluster_labels is not None:
            upsampled_labels = torch.nn.functional.interpolate(
                cluster_labels.float().unsqueeze(1),  # [B, 1, H, W]
                size=target_size,
                mode='nearest'
            ).squeeze(1).long()  # [B, H', W']
        
        # use matplotlib's Spectral_r color map
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('Spectral_r')
        
        # apply color map to each attention map
        H_up, W_up = upsampled_attns.shape[1:]
        stacked_att_img = torch.zeros((B, 3, H_up, W_up), device=upsampled_attns.device)
        for i in range(B):
            att_np = upsampled_attns[i].cpu().numpy()
            colored_att = cmap(att_np)[:, :, :3]  # remove alpha channel
            colored_att_torch = torch.from_numpy(colored_att).float().permute(2, 0, 1).to(upsampled_attns.device)
            stacked_att_img[i] = colored_att_torch

        # calculate mask
        stacked_mask = (upsampled_attns > adaptive_multiotsu_variance(upsampled_attns.cpu().numpy()))

        if cluster_labels is not None:
            import matplotlib.pyplot as plt
            num_clusters = upsampled_labels.max().item() + 1
            colors = plt.cm.tab20(np.linspace(0, 1, num_clusters))[:, :3]
            colors = torch.from_numpy(colors).float().to(upsampled_labels.device)
            
            stacked_mask_img = torch.zeros((B, 3, H_up, W_up), device=upsampled_labels.device)
            for i in range(num_clusters):
                mask = (upsampled_labels == i) & stacked_mask 
                mask = mask.unsqueeze(1)  # [B, 1, H', W']
                stacked_mask_img += mask * colors[i].view(1, 3, 1, 1)
        else:
            stacked_mask_img = stacked_mask.unsqueeze(1).expand(-1, 3, -1, -1)  # [B, 3, H', W']

        # create grid layout  
        grid_size = int(math.ceil(math.sqrt(B)))
        # for stacked_att and cluster_map create grid
        grid_att = torchvision.utils.make_grid(stacked_att_img, nrow=grid_size, padding=2, normalize=False)
        grid_cluster = torchvision.utils.make_grid(stacked_mask_img, nrow=grid_size, padding=2, normalize=False)
        # concatenate two grids in vertical direction
        final_grid = torch.cat([grid_att, grid_cluster], dim=1)
        torchvision.utils.save_image(final_grid, os.path.join(save_folder, f'0_{save_name}_fused.png'))

        # vis
        fused_save_folder = os.path.join(save_folder, f'0_{save_name}_fused')
        os.makedirs(fused_save_folder, exist_ok=True)

        # save video
        #TODO implement ffmpeg properly
        if B > 0 and False:
            # create frames directory for stacked_att_img
            frames_att_dir = os.path.join(fused_save_folder, 'frames_att')
            os.makedirs(frames_att_dir, exist_ok=True)
            
            for i in range(B):
                att_frame = stacked_att_img[i].cpu().numpy().transpose(1, 2, 0)  # convert to HWC format
                frame = (att_frame * 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert to BGR format
                cv2.imwrite(f'{frames_att_dir}/frame_{i:04d}.png', frame)
            
            # use ffmpeg to generate video, frame rate set to 24
            video_att_path = os.path.join(fused_save_folder, f'0_{save_name}_att_video.mp4')
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{frames_att_dir}/frame_%04d.png" '
                    f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                    f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                    f'-movflags +faststart -b:v 5000k "{video_att_path}"')

            # create frames directory for stacked_mask_img
            frames_mask_dir = os.path.join(fused_save_folder, 'frames_mask')
            os.makedirs(frames_mask_dir, exist_ok=True)
            
            for i in range(B):
                mask_frame = stacked_mask_img[i].cpu().numpy().transpose(1, 2, 0)  # convert to HWC format
                frame = (mask_frame * 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert to BGR format
                cv2.imwrite(f'{frames_mask_dir}/frame_{i:04d}.png', frame)
            
            # # use ffmpeg to generate video, frame rate set to 24
            # video_mask_path = os.path.join(fused_save_folder, f'0_{save_name}_mask_video.mp4')
            # os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{frames_mask_dir}/frame_%04d.png" '
            #         f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
            #         f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
            #         f'-movflags +faststart -b:v 5000k "{video_mask_path}"')