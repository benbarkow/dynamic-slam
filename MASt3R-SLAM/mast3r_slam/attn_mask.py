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
    def __init__(self, all_results, edges):
        self.edges = edges
        self.pred1 = {}
        self.pred2 = {}

        self.pred1["cross_atten_maps_k"] = torch.stack([res[0]["cross_atten_maps_k"].squeeze(0) for res in all_results])
        self.pred2["cross_atten_maps_k"] = torch.stack([res[1]["cross_atten_maps_k"].squeeze(0) for res in all_results])

        self.pred1["pts3d"] = torch.stack([res[0]["pts3d"].squeeze(0) for res in all_results])
        self.pred2["pts3d_in_other_view"] = torch.stack([res[1]["pts3d_in_other_view"].squeeze(0) for res in all_results])
        self.pred1["match_feature"] = torch.stack([res[0]["match_feature"].squeeze(0) for res in all_results])
    
        self.pred_i = NoGradParamDict(
            {ij: self.pred1["pts3d"][n] for n, ij in enumerate(self.str_edges)}
        )
        self.pred_j = NoGradParamDict(
            {ij: self.pred2["pts3d_in_other_view"][n] for n, ij in enumerate(self.str_edges)}
        )
        # Note: The original code used pts3d for both imshapes arguments. Preserving that here.
        self.imshapes = get_imshapes(self.edges, self.pred1["pts3d"], self.pred2["pts3d_in_other_view"])
        
    def get_dynamic_map(self, id):
        upsampled_attns = torch.nn.functional.interpolate(self.refined_dynamic_map.unsqueeze(-1).permute(0, 3, 1, 2), \
                                                size=self.imshapes[0], mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(-1) # align_corners=False
        upsampled_mask = (upsampled_attns > self.adaptive_multiotsu_variance(upsampled_attns.cpu().numpy()))
        # self.vis_attention_masks(upsampled_attns, save_folder="test/", save_name='test', id=id)
        # self.save_simple(upsampled_mask, save_folder="test/", save_name='upsampled', id=id)
        return upsampled_mask

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
            # print("att_maps min/max:", att_maps_min, att_maps_max)
            # print("att_maps contains nan:", torch.isnan(att_maps).any())
            att_maps_normalized = (att_maps - att_maps_min) / (
                att_maps_max - att_maps_min + 1e-6
            )
            # average channel
            att_maps_fused = att_maps_normalized.mean(dim=-1)  # B, H, W
            # normalize
            att_maps_fused_min = att_maps_fused.min()
            att_maps_fused_max = att_maps_fused.max()
            # print("att_maps_fused min/max:", att_maps_fused_min, att_maps_fused_max)
            # print("att_maps_fused contains nan:", torch.isnan(att_maps_fused).any())
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
        
        #create dynamic mask
        dynamic_map = (1-self.cross_att_k_i_mean_fused) * self.cross_att_k_i_var_fused * self.cross_att_k_j_mean_fused * (1-self.cross_att_k_j_var_fused)
        dynamic_map_min = dynamic_map.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0] # B, 1, 1
        dynamic_map_max = dynamic_map.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] # B, 1, 1
        self.dynamic_map = (dynamic_map - dynamic_map_min) / (dynamic_map_max - dynamic_map_min + 1e-6)

        pred1_feat = pred1['match_feature']
        feat_i = NoGradParamDict({ij: nn.Parameter(pred1_feat[n], requires_grad=False) for n, ij in enumerate(self.str_edges)})
        stacked_feat_i = [feat_i[k] for k in self.str_edges]
        stacked_feat = [None] * len(self.imshapes)
        for i, ei in enumerate(torch.tensor([i for i, j in self.edges])):
            stacked_feat[ei]=stacked_feat_i[i]
        self.stacked_feat = torch.stack(stacked_feat).float().detach()
        
        self.refined_dynamic_map, self.dynamic_map_labels = self.cluster_attention_maps(self.stacked_feat, self.dynamic_map, n_clusters=64)
   

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
        
    def save_attention_maps(self, id="0", save_folder='demo_tmp/attention_vis',):
        self.vis_attention_masks(1-self.cross_att_k_i_mean_fused, save_folder=save_folder, save_name='cross_att_k_i_mean', id=id)
        self.vis_attention_masks(self.cross_att_k_i_var_fused, save_folder=save_folder, save_name='cross_att_k_i_var', id=id)
        self.vis_attention_masks(1-self.cross_att_k_j_mean_fused, save_folder=save_folder, save_name='cross_att_k_j_mean', id=id)
        self.vis_attention_masks(self.cross_att_k_j_var_fused, save_folder=save_folder, save_name='cross_att_k_j_var', id=id)
        self.vis_attention_masks(self.dynamic_map, save_folder=save_folder, save_name='dynamic_map', id=id)
        self.vis_attention_masks(self.refined_dynamic_map, save_folder=save_folder, save_name='refined_dynamic_map', id=id)
        self.vis_attention_masks(self.refined_dynamic_map, save_folder=save_folder, save_name='refined_dynamic_map_labels', \
                            cluster_labels=self.dynamic_map_labels, id=id)
    def adaptive_multiotsu_variance(self, img, verbose=False):
        # print(img)
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
    
    @torch.no_grad()
    def save_simple(self, maps, save_folder, save_name, id):
        # create grid layout  
        B, H, W = maps.shape
        masks = maps.float().unsqueeze(1)

        # Optional: Invert if you want white background and black mask
        # masks = 1.0 - masks

        # Create grid
        grid_size = int(math.ceil(math.sqrt(B)))
        grid_img = torchvision.utils.make_grid(masks, nrow=grid_size, padding=2, normalize=False)

        # Save image
        save_path = os.path.join(f'{save_folder}/frame_{id}/')
        os.makedirs(save_path, exist_ok=True)
        torchvision.utils.save_image(grid_img, os.path.join(save_path, f'{id}_{save_name}_binary.png'))

    @torch.no_grad()
    def vis_attention_masks(self, attns_fused, save_folder='demo_tmp/attention_vis', save_name='attention_channels_all_frames', cluster_labels=None, id="0"):

        

        B, H, W = attns_fused.shape
        # ensure self.imshape exists, otherwise use the original size
        target_size = getattr(self, 'imshape', (384, 512))
        # print(target_size)
        
        # upsample the attention maps
        # print("upsampled before", attns_fused.shape)
        # print("nones", torch.isnan(attns_fused).any())
        upsampled_attns = torch.nn.functional.interpolate(
            attns_fused.unsqueeze(1),  # [B, 1, H, W]
            size=target_size, 
            mode='nearest'
        ).squeeze(1)  # [B, H', W']
        
        # print("upsampled after", upsampled_attns.shape)
        # print("upsampled after", upsampled_attns)
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
        stacked_mask = (upsampled_attns > self.adaptive_multiotsu_variance(upsampled_attns.cpu().numpy()))

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
        os.makedirs(os.path.join(f'{save_folder}/frame_{id}/'), exist_ok=True)
        torchvision.utils.save_image(final_grid, os.path.join(f'{save_folder}/frame_{id}/', f'{id}_{save_name}_fused.png'))

        # # vis
        # fused_save_folder = os.path.join(save_folder, f'0_{save_name}_fused')
        # os.makedirs(fused_save_folder, exist_ok=True)

    @torch.no_grad()
    def cluster_attention_maps(self, feature, dynamic_map, n_clusters=64):
        """use KMeans to cluster the attention maps using feature
        
        Args:
            feature: encoder feature [B,H,W,C]
            dynamic_map: dynamic_map feature [B,H,W]
            n_clusters: number of clusters
            
        Returns:
            normalized_map: normalized cluster map [B,H,W]
            cluster_labels: reshaped cluster labels [B,H,W]
        """
        # data preprocessing
        B, H, W, C = feature.shape
        feature_np = feature.cpu().numpy()
        flattened_feature = feature_np.reshape(-1, C)
        
        # KMeans clustering
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(flattened_feature)
        
        # calculate the average dynamic score for each cluster
        dynamic_map_np = dynamic_map.cpu().numpy()
        flattened_dynamic = dynamic_map_np.reshape(-1)
        cluster_dynamic_scores = np.zeros(n_clusters)
        for i in range(n_clusters):
            cluster_mask = (cluster_labels == i)
            cluster_dynamic_scores[i] = np.mean(flattened_dynamic[cluster_mask])
        
        # map the cluster labels to the dynamic score
        cluster_map = cluster_dynamic_scores[cluster_labels]
        normalized_map = cluster_map.reshape(B, H, W)

        # reshape cluster_labels
        reshaped_labels = cluster_labels.reshape(B, H, W)
        
        # convert to torch tensor
        normalized_map = torch.from_numpy(normalized_map).float()
        cluster_labels = torch.from_numpy(reshaped_labels).long()
        
        normalized_map_min = normalized_map.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        normalized_map_max = normalized_map.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        normalized_map = (normalized_map - normalized_map_min) / (normalized_map_max - normalized_map_min + 1e-6)

        return normalized_map, cluster_labels