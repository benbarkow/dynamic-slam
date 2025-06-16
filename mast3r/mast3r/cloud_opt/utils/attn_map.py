import torch
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

class AttentionMaskGenerator:

    def __init__(
            self,
            res,
            view1,
            view2
    ):
        self.pred1 = res[0]
        self.pred2 = res[1]

        # if not isinstance(view1['idx'], list):
        #     view1['idx'] = view1['idx'].tolist()
        # if not isinstance(view2['idx'], list):
        #     view2['idx'] = view2['idx'].tolist()
        # self.edges = [(int(i), int(j)) for i, j in zip(view1['idx'], view2['idx'])]
        self.edges = [(0, 1)]

        pred1_pts = self.pred1['pts3d']
        pred2_pts = self.pred2['pts3d_in_other_view']
        self.pred_i = NoGradParamDict({ij: pred1_pts[n] for n, ij in enumerate(self.str_edges)})
        self.pred_j = NoGradParamDict({ij: pred2_pts[n] for n, ij in enumerate(self.str_edges)})
        self.imshapes = get_imshapes(self.edges, pred1_pts, pred2_pts)
        print(self.imshapes)
        pass

    def str_edges(self):
        return [edge_str(i, j) for i, j in self.edges]
      
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