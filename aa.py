# modify from https://github.com/mit-han-lab/bevfusion
import copy
from typing import List, Tuple

import numpy as np
import torch
import random
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet.models.task_modules import (AssignResult, PseudoSampler,
                                       build_assigner, build_bbox_coder,
                                       build_sampler)
from mmdet.models.utils import multi_apply
from mmengine.structures import InstanceData
from torch import nn

from mmdet3d.models import circle_nms, draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models.dense_heads.centerpoint_head import SeparateHead
from mmdet3d.models.layers import nms_bev
from mmdet3d.registry import MODELS
from mmdet3d.structures import xywhr2xyxyr
from .encoder_utils import LocalContextAttentionBlock_BEV, ConvBNReLU
from timm.models.layers import trunc_normal_
import math
from .deformable_transformer import build_deforamble_transformer
from .deformable_utils.position_encoding import PositionEmbeddingSine
from .utils import NestedTensor
import cv2
import shutil
from mmdet3d.models.layers.fusion_layers import apply_3d_transformation
from .depth_map_utils import fill_in_multiscale
from .encoder_unibev_detr_pts import PtsEncoder
from mmengine.model import BaseModule

def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y

class LearnedPositionalEncoding(BaseModule):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(LearnedPositionalEncoding, self).__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).permute(2, 0,
                            1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos


    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str
    

@MODELS.register_module()
class DeformableTransformer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.mask_freq = kwargs.pop("mask_freq")
        self.mask_ratio = kwargs.pop("mask_ratio")
        
        self.mask_img=kwargs.pop("mask_img", False)
        self.mask_pts=kwargs.pop("mask_pts", False)
        self.fusion_method=kwargs.get('fusion_method',False)
        self.mask_method = kwargs.get('mask_method', 'point')
        if 'patch' in self.mask_method:
            self.patch_cfg = kwargs.get('patch_cfg', None)
        if kwargs.get('residual', False):
            self.residual = kwargs.pop("residual")
        else:
            self.residual = False
        if kwargs.get('loss_weight', False):
            self.loss_weight = kwargs.pop("loss_weight")
        else:
            self.loss_weight = 1
        if self.mask_pts:
            self.model = build_deforamble_transformer(**kwargs)
        self.pts_bev_encoder = PtsEncoder(**kwargs['pts_encoder'])
        self.img_bev_encoder = PtsEncoder(**kwargs['img_encoder'])
        pts_channels = 256
        img_channels = 80     
        if self.mask_img:
            img_kwargs = copy.deepcopy(kwargs)
            img_kwargs['d_model'] = img_channels
            img_kwargs['num_encoder_layers'] = kwargs.get('num_img_encoder_layers',False)
            self._model = build_deforamble_transformer(**img_kwargs)
        self.conv = nn.Sequential(nn.Conv2d(
                pts_channels+img_channels, pts_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(pts_channels),
            nn.ReLU(True))
        if self.mask_pts:
            self.position_embedding = PositionEmbeddingSine(
                num_pos_feats= pts_channels // 2, normalize=True)
            self.input_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(pts_channels, pts_channels, kernel_size=1),
                        nn.GroupNorm(32, pts_channels),
                    )])        
        if self.mask_img:
            self._position_embedding = PositionEmbeddingSine(
                num_pos_feats= img_channels // 2, normalize=True)
            self._input_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(img_channels, img_channels, kernel_size=1),
                        nn.GroupNorm(10, img_channels),
                    )])
        self.num_cross_attention_layers = kwargs.get('num_cross_attention_layers', False)
        self._nheads = kwargs.get('_nheads', False)
        if self.num_cross_attention_layers or self._nheads or self.fusion_method:
            if self.mask_pts:
                self.target_proj = nn.ModuleList([
                        nn.Sequential(
                            nn.Conv2d(img_channels, img_channels, kernel_size=1),
                            nn.GroupNorm(10, img_channels),
                        )]) 
            if self.mask_img:
                self._target_proj = nn.ModuleList([
                        nn.Sequential(
                            nn.Conv2d(pts_channels, pts_channels, kernel_size=1),
                            nn.GroupNorm(32, pts_channels),
                        )]) 
        if self.mask_pts:
            self.pts_mask_tokens = nn.Parameter(torch.zeros(1, 1, pts_channels))
            self.pred = nn.Conv2d(pts_channels, pts_channels, kernel_size=1)
        if self.mask_img:
            self.img_mask_tokens = nn.Parameter(torch.zeros(1, 1, img_channels))
            self._pred = nn.Conv2d(img_channels, img_channels, kernel_size=1)
            self.linear = nn.Conv2d(img_channels, img_channels, kernel_size=1)
        if self.residual == 'concat':
            self.P_integration = ConvBNReLU(2 * pts_channels, pts_channels, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)
        self.pts_embedding = nn.Embedding(
            180 * 180, 256)
        self.img_embedding = nn.Embedding(
            180 * 180, 80)
        self.pts_positional_encoding = LearnedPositionalEncoding(
            num_feats=128, row_num_embed=180, col_num_embed=180)
        self.img_positional_encoding = LearnedPositionalEncoding(
            num_feats=40, row_num_embed=180, col_num_embed=180)
        self.pts_level_embeds = nn.Parameter(torch.Tensor(
                1, 256))
        self.img_level_embeds = nn.Parameter(torch.Tensor(
                1, 80))
        
        self.initialize_weights()
        
    def initialize_weights(self):
        if self.mask_pts:
            for proj in self.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)
            if self.num_cross_attention_layers or self._nheads or self.fusion_method:
                for _proj in self.target_proj:
                    nn.init.xavier_uniform_(_proj[0].weight, gain=1)
                    nn.init.constant_(_proj[0].bias, 0)
        if self.mask_img:
            for __proj in self._input_proj:
                nn.init.xavier_uniform_(__proj[0].weight, gain=1)
                nn.init.constant_(__proj[0].bias, 0)
            if self.num_cross_attention_layers or self._nheads or self.fusion_method:
                for _proj_ in self._target_proj:
                    nn.init.xavier_uniform_(_proj_[0].weight, gain=1)
                    nn.init.constant_(_proj_[0].bias, 0)
        if self.mask_pts:
            torch.nn.init.normal_(self.pts_mask_tokens, std=.02)
        if self.mask_img:
            torch.nn.init.normal_(self.img_mask_tokens, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def random_patch_masking(self, _x):
        x = _x.clone()
        B, D, W, H = x.shape
        if D == 256:
            mask_tokens = self.pts_mask_tokens.permute(2, 0, 1)
        elif D == 80:
            mask_tokens = self.img_mask_tokens.permute(2, 0, 1)

        mask = torch.zeros([B, W, H], device=x.device)
        for b in range(B):
            # pick size and number of patch
            l_min, l_max = self.patch_cfg.len_min, self.patch_cfg.len_max
            rand_s_l = torch.randint(l_min, l_max, (500,))
            rand_s_l_sq = torch.cumsum(torch.square(rand_s_l), dim=0)
            rand_n = (rand_s_l_sq > (W*H*self.mask_ratio)).nonzero()[0][0]

            # masking patch
            rand_x_l, rand_y_l = torch.randint(0, W, (rand_n, 1)), torch.randint(0, H, (rand_n, 1))
            for rand_s, rand_x, rand_y in zip(rand_s_l, rand_x_l, rand_y_l):
                x[b][:, rand_x:rand_x+rand_s, rand_y:rand_y+rand_s] = mask_tokens
                mask[b][rand_x:rand_x+rand_s, rand_y:rand_y+rand_s] = 1

        return x, mask.flatten(1)
    def visualize_feat(self, bev_feat, idx):
        feat = bev_feat.cpu().detach().numpy()
        min = feat.min()
        max = feat.max()
        image_features = (feat-min)/(max-min)
        image_features = (image_features*255)
        max_image_feature = np.max(np.transpose(image_features.astype("uint8"),(1,2,0)),axis=2)
        max_image_feature = cv2.applyColorMap(max_image_feature,cv2.COLORMAP_JET)
        cv2.imwrite(f"max_{idx}.jpg",max_image_feature)

    def _pre_process_pts_feats(self, mlvl_pts_feats, bev_queries):
        ## process multi-level points features
        pts_feat_flatten = []
        pts_spatial_shapes = []
        lvl = 0
        feat = mlvl_pts_feats
        bs, c, h, w = feat.shape
        pts_spatial_shape = (h, w)
        feat = feat.flatten(2).permute(0, 2, 1)
        # print(' feat size:', feat.size()) # [2, 40000, 512]
        if bev_queries.shape[-1] == 80:
            feat = feat + self.img_level_embeds[None, lvl:lvl + 1, :].to(feat.dtype)
        else:    
            feat = feat + self.pts_level_embeds[None, lvl:lvl + 1, :].to(feat.dtype)
        pts_spatial_shapes.append(pts_spatial_shape)
        pts_feat_flatten.append(feat)

        pts_feat_flatten = torch.cat(pts_feat_flatten, 2)
        pts_spatial_shapes = torch.as_tensor(pts_spatial_shapes, dtype=torch.long, device=bev_queries.device)
        pts_level_start_index = torch.cat((pts_spatial_shapes.new_zeros((1,)), pts_spatial_shapes.prod(1).cumsum(0)[:-1]))

        pts_feat_flatten = pts_feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)

        return pts_feat_flatten, pts_spatial_shapes, pts_level_start_index

    
    def forward_single(self, inputs: List[torch.Tensor], fg_bg_mask_list, sensor_list=None) -> torch.Tensor:
        if self.residual:
            residual_pts = inputs[1].clone()
            residual_img = inputs[0].clone()
    
        prob = np.random.uniform()
        _mask = prob < self.mask_freq
        
        dtype = inputs[0].dtype
        bev_queries_img = self.img_embedding.weight.to(dtype)
        bev_queries_pts = self.pts_embedding.weight.to(dtype)
        bev_queries_img = bev_queries_img.unsqueeze(1).repeat(1, 1, 1)
        bev_queries_pts = bev_queries_pts.unsqueeze(1).repeat(1, 1, 1)
        
        ## points
        if _mask and inputs[1].requires_grad and self.mask_pts:
            if self.mask_method == 'random_patch':
                pts_target = inputs[1].flatten(2).transpose(1, 2).clone()
                src, pts_mask = self.random_patch_masking(inputs[1])
        else:
            src = inputs[1].clone()
        device=inputs[0].device
        bev_mask = torch.zeros((1, 180, 180),
                               device=device).to(dtype)
        pts_bev_pos = self.pts_positional_encoding(bev_mask).to(dtype)
        img_bev_pos = self.img_positional_encoding(bev_mask).to(dtype)
        pts_bev_pos = pts_bev_pos.flatten(2).permute(2, 0, 1)
        img_bev_pos = img_bev_pos.flatten(2).permute(2, 0, 1)
        if self.mask_pts:
            pts_feat_flatten, pts_spatial_shapes, pts_level_start_index = self._pre_process_pts_feats(src, bev_queries_pts)
            # self.visualize_feat(bev_queries_pts.squeeze().transpose(0,1).view(256,180,180),'bev_query')
            # self.visualize_feat(pts_feat_flatten.squeeze().transpose(0,1).view(256,180,180),'pts_feat_flatten')
            pts_bev_embed = self.pts_bev_encoder(
                bev_queries_pts,
                pts_feat_flatten,
                pts_feat_flatten,
                bev_h=180,
                bev_w=180,
                bev_pos=pts_bev_pos,
                spatial_shapes=pts_spatial_shapes,
                level_start_index=pts_level_start_index) # encoder.batch_first = True: (bs, bev_h*bev_w, embed_dims)
            # self.visualize_feat(pts_bev_embed.squeeze().transpose(0,1).view(256,180,180),'pts_bev_embed')
            pts_bev_embed = pts_bev_embed.transpose(1,2).view(1,256,180,180)
            # pts_feat = self.pred(pts_bev_embed)
            pts_feat = pts_bev_embed.contiguous()
        if _mask and inputs[1].requires_grad and self.mask_pts:
            pts_feat = pts_feat.flatten(2).transpose(1, 2)         
            if fg_bg_mask_list is not None:
                device=pts_feat.device
                fg_mask, bg_mask = fg_bg_mask_list
                fg_mask, bg_mask = fg_mask.to(device), bg_mask.to(device)
                fg_mask, bg_mask = fg_mask.flatten(2).transpose(1,2), bg_mask.flatten(2).transpose(1,2)
                fg_loss = (pts_feat - pts_target) ** 2 * fg_mask
                bg_loss = (pts_feat - pts_target) ** 2 * bg_mask
                bg_loss = 0.2 * bg_loss
                bg_loss = bg_loss.mean(dim=-1)  # [N, L], mean loss per patch
                fg_loss = fg_loss.mean(dim=-1)  # [N, L], mean loss per patch
                if (fg_mask.squeeze()*pts_mask).sum() == 0:
                    fg_loss = (fg_loss * pts_mask).sum()
                else:
                    fg_loss = (fg_loss * pts_mask).sum() / (fg_mask.squeeze()*pts_mask).sum()  # mean loss on removed patches
                bg_loss = (bg_loss * pts_mask).sum() / (bg_mask.squeeze()*pts_mask).sum()  # mean loss on removed patches
                pts_fg_loss = self.loss_weight * fg_loss
                pts_bg_loss = self.loss_weight * bg_loss
                pts_feat = pts_feat.transpose(1,2).view(1,256,180,180)
            else:
                loss = (pts_feat - pts_target) ** 2
                loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
                loss = (loss * pts_mask).sum() / pts_mask.sum()  # mean loss on removed patches
                pts_loss = self.loss_weight * loss
                pts_feat = pts_feat.transpose(1,2).view(1,256,180,180)
        if self.residual == 'sum':
            pts_feat += residual_pts
        ## img        
        if _mask and inputs[0].requires_grad and self.mask_img:
            if self.mask_method == 'random_patch':
                img_target = inputs[0].flatten(2).transpose(1, 2).clone()
                _src, img_mask = self.random_patch_masking(inputs[0])
        else:
            _src = inputs[0].clone()
        
        if self.mask_img:
            img_feat_flatten, img_spatial_shapes, img_level_start_index = self._pre_process_pts_feats(_src, bev_queries_img)
            # self.visualize_feat(bev_queries_img.squeeze().transpose(0,1).view(80,180,180),'bev_query_img')
            # self.visualize_feat(img_feat_flatten.squeeze().transpose(0,1).view(80,180,180),'img_feat_flatten')
            img_bev_embed = self.img_bev_encoder(
                bev_queries_img,
                img_feat_flatten,
                img_feat_flatten,
                bev_h=180,
                bev_w=180,
                bev_pos=img_bev_pos,
                spatial_shapes=img_spatial_shapes,
                level_start_index=img_level_start_index) # encoder.batch_first = True: (bs, bev_h*bev_w, embed_dims)
            # self.visualize_feat(img_bev_embed.squeeze().transpose(0,1).view(80,180,180),'img_bev_embed')
            img_bev_embed = img_bev_embed.transpose(1,2).view(1,80,180,180)
            # img_feat = self._pred(img_bev_embed)
            img_feat = img_bev_embed.contiguous()
        
        if _mask and inputs[0].requires_grad and self.mask_img:
            img_feat = img_feat.flatten(2).transpose(1, 2)         
            if fg_bg_mask_list is not None:
                device=img_feat.device
                fg_mask, bg_mask = fg_bg_mask_list
                fg_mask, bg_mask = fg_mask.to(device), bg_mask.to(device)
                fg_mask, bg_mask = fg_mask.flatten(2).transpose(1,2), bg_mask.flatten(2).transpose(1,2)
                fg_loss = (img_feat - img_target) ** 2 * fg_mask
                bg_loss = (img_feat - img_target) ** 2 * bg_mask
                bg_loss = 0.2 * bg_loss
                bg_loss = bg_loss.mean(dim=-1)  # [N, L], mean loss per patch
                fg_loss = fg_loss.mean(dim=-1)  # [N, L], mean loss per patch
                if (fg_mask.squeeze()*img_mask).sum() == 0:
                    fg_loss = (fg_loss * img_mask).sum()
                else:
                    fg_loss = (fg_loss * img_mask).sum() / (fg_mask.squeeze()*img_mask).sum()  # mean loss on removed patches
                bg_loss = (bg_loss * img_mask).sum() / (bg_mask.squeeze()*img_mask).sum()  # mean loss on removed patches
                img_fg_loss = self.loss_weight * fg_loss
                img_bg_loss = self.loss_weight * bg_loss
                img_feat = img_feat.transpose(1,2).view(1,80,180,180)
            else:
                loss = (img_feat - img_target) ** 2
                loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
                loss = (loss * img_mask).sum() / img_mask.sum()  # mean loss on removed patches
                img_loss = self.loss_weight * loss
                img_feat = img_feat.transpose(1,2).view(1,80,180,180)
        if self.residual == 'sum':
                img_feat += residual_img
        if _mask and inputs[0].requires_grad:
            loss_list = dict()
            if fg_bg_mask_list is not None:
                if self.mask_pts:
                    loss_list['pts_fg_loss'] = pts_fg_loss
                    loss_list['pts_bg_loss'] = pts_bg_loss
                if self.mask_img:
                    loss_list['img_fg_loss'] = img_fg_loss
                    loss_list['img_bg_loss'] = img_bg_loss
                return self.conv(torch.cat([img_feat, pts_feat], dim=1)), loss_list
            else:
                if self.mask_pts:
                    loss_list['pts_loss'] = pts_loss
                if self.mask_img:
                    loss_list['img_loss'] = img_loss
                return self.conv(torch.cat([img_feat, pts_feat], dim=1)), loss_list
        return self.conv(torch.cat([img_feat, pts_feat], dim=1)), False

    def forward(
        self, 
        inputs: List[torch.Tensor], 
        fg_bg_mask_list, 
        sensor_list=None,
        batch_input_metas=None,
    ) -> torch.Tensor:
        batch = len(batch_input_metas)
        loss_list, feat_list = [], []
        loss_flag = False
        for b in range(batch):
            inputs_ = [inputs[0][b].unsqueeze(0), inputs[1][b].unsqueeze(0)]
            if fg_bg_mask_list is not None:
                fg_bg_mask_list_ = [fg_bg_mask_list[0][b].unsqueeze(0),
                                    fg_bg_mask_list[1][b].unsqueeze(0)]
                sensor_list_ = [True, True]
            else:
                fg_bg_mask_list_ = None
                sensor_list_ = None
            feat, loss_l = self.forward_single(inputs_, fg_bg_mask_list_, sensor_list_)
            feat_list.append(feat)
            loss_list.append(loss_l)
            if loss_l:
                loss_flag = True

        for b in range(batch):
            if loss_list[b]:
                if batch_input_metas[b]['smt_number'] != 2:
                    for key in list(loss_list[b].keys()):
                        loss_list[b][key] *= 0.

        if loss_flag:
            if batch == 2:
                if loss_list.count(False) == 0:
                    #  print(0)
                    for key in list(loss_list[0].keys()):
                        loss_list[0][key] += loss_list[1][key]
                        loss_list[0][key] /= 2
                else:
                    #  print(1)
                    idx = loss_list.index(False)
                    loss_list.pop(idx)
            loss_list = loss_list[0]
        else:
            #  print(2)
            loss_list = False

        return torch.cat(feat_list, dim=0), loss_list

class BEVWarp(nn.Module):
    
    def __init__(self):
        super().__init__()

    def visualize_feat(self, bev_feat, idx):
        feat = bev_feat.cpu().detach().numpy()
        min = feat.min()
        max = feat.max()
        image_features = (feat-min)/(max-min)
        image_features = (image_features*255)
        max_image_feature = np.max(np.transpose(image_features.astype("uint8"),(1,2,0)),axis=2)
        max_image_feature = cv2.applyColorMap(max_image_feature,cv2.COLORMAP_JET)
        cv2.imwrite(f"max_{idx}.jpg",max_image_feature)
    def visualize_pts(self, pts_2d, mask, pts_idx, idx):
        Img = np.zeros((900, 1600, 3), np.uint8)
        i_coor = pts_2d[pts_idx][mask[pts_idx].squeeze(-1)]
        for coor in i_coor:
            cv2.circle(Img,(int(coor[0].item()),int(coor[1].item())), 5,(0,0,255))
        cv2.imwrite(f"pts_{idx}.jpg",Img)


    def project_lidar_to_cam(self, pts_3d_, img_meta):
        img_aug_matrix = pts_3d_.new_tensor(img_meta['img_aug_matrix'])
        if 'lidar_aug_matrix' in img_meta:
            lidar_aug_matrix = pts_3d_.new_tensor(img_meta['lidar_aug_matrix'])
        else:
            lidar_aug_matrix = torch.eye(4).cuda()
        lidar2image = pts_3d_.new_tensor(img_meta['lidar2img'])

        pts_3d = pts_3d_.clone()
        # inverse aug
        pts_3d -= lidar_aug_matrix[:3, 3]
        pts_3d = torch.inverse(lidar_aug_matrix[:3, :3]).matmul(pts_3d.transpose(1, 0))
        # lidar2image
        pts_3d = lidar2image[:, :3, :3].matmul(pts_3d)
        pts_3d += lidar2image[:, :3, 3].reshape(-1, 3, 1)
        # get 2d coords
        dist = pts_3d[:, 2, :]
        pts_3d[:, 2, :] = torch.clamp(pts_3d[:, 2, :], 1e-5, 1e5)
        pts_3d[:, :2, :] /= pts_3d[:, 2:3, :]
        # imgaug
        pts_3d = img_aug_matrix[:, :3, :3].matmul(pts_3d)
        pts_3d += img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
        #pts_3d = pts_3d[:, :2, :].transpose(1, 2)
        pts_3d = pts_3d[:, :, :].transpose(1, 2)

        return pts_3d


    def forward(self, lidar_feats, img_feats, img_metas, pts_metas, batch_index, img=None, points=None):
        batch_size, num_views, I_C, I_H, I_W = img_feats.shape
        # lidar2img = []
        # for img_meta in img_metas:
        #     lidar2img.append(img_meta['lidar2img'])
        # lidar2img = np.asarray(lidar2img)
        lidar2img = np.asarray(img_metas['lidar2img'])
        lidar2img = img_feats.new_tensor(lidar2img)
        img2lidar = torch.inverse(lidar2img)
        pts = pts_metas['pts']
        decorated_img_feats = []
        b = batch_index
        img_feat = img_feats[0]
        ori_H, ori_W = 256, 704
        pts_3d = pts[b][...,:3]
        pts_2d = self.project_lidar_to_cam(pts_3d, img_metas)
        depth = pts_2d[..., 2:3]
        proj_x = (pts_2d[...,0:1] / ori_W - 0.5) * 2
        proj_y = (pts_2d[...,1:2] / ori_H - 0.5) * 2
        mask = (proj_x > -1.0) & (proj_x < 1.0) & (proj_y > -1.0) & (proj_y < 1.0)
        mask = torch.nan_to_num(mask)
        
        
        # feat = img[6*batch_index].cpu().detach().numpy()
        # min = feat.min()
        # max = feat.max()
        # image_features = (feat-min)/(max-min)
        # image_features = (image_features*255)
        # max_image_feature = np.max(np.transpose(image_features.astype("uint8"),(1,2,0)),axis=2)
        # Img = cv2.applyColorMap(max_image_feature,cv2.COLORMAP_JET)
        # i_coor = pts_2d[0][mask[0, :, 0]]
        # for coor in i_coor:
        #     cv2.circle(Img,(int(coor[0].item()),int(coor[1].item())), 1,(0,0,255))
        # cv2.imwrite('test.jpg', Img)

        depth_map = img_feat.new_zeros(num_views, I_H, I_W)
        for i in range(num_views):
            depth_map[i, (pts_2d[i,mask[i,:,0],1]/ori_H*I_H).long(), (pts_2d[i,mask[i,:,0],0]/ori_W*I_W).long()] = depth[i,mask[i,:,0],0]
        fill_type = 'multiscale'
        extrapolate = False
        blur_type = 'bilateral'
        for i in range(num_views):
            final_depths, _ = fill_in_multiscale(
                            depth_map[i].detach().cpu().numpy(), extrapolate=extrapolate, blur_type=blur_type,
                            show_process=False)
            depth_map[i] = depth_map.new_tensor(final_depths)
        xs = torch.linspace(0, ori_W - 1, I_W, dtype=torch.float32).to(depth_map.device).view(1, 1, I_W).expand(num_views, I_H, I_W)
        ys = torch.linspace(0, ori_H - 1, I_H, dtype=torch.float32).to(depth_map.device).view(1, I_H, 1).expand(num_views, I_H, I_W)
        xyd = torch.stack((xs, ys, depth_map, torch.ones_like(depth_map)), dim = -1)
        xyd [..., 0] *= xyd [..., 2]
        xyd [..., 1] *= xyd [..., 2]
        xyz = img2lidar.view(num_views,1,1,4,4).matmul(xyd.unsqueeze(-1)).squeeze(-1)[...,:3] #(6,112,200,3)
        xyz = apply_3d_transformation(xyz.view(num_views*I_H*I_W, 3), 'LIDAR', img_metas, reverse=False).view(num_views, I_H, I_W, 3).detach()
        pc_range = xyz.new_tensor([-54, -54, -5, 54, 54, 3])  #TODO: fix it to support other outdoor dataset!!!
        lift_mask = (xyz[...,0] > pc_range[0]) & (xyz[...,1] > pc_range[1]) & (xyz[...,2] > pc_range[2])\
                    & (xyz[...,0] < pc_range[3]) & (xyz[...,1] < pc_range[4]) & (xyz[...,2] < pc_range[5])
        xy_bev = (xyz[...,0:2] - pc_range[0:2]) / (pc_range[3:5] - pc_range[0:2])
        xy_bev = (xy_bev - 0.5) * 2
        decorated_img_feat = F.grid_sample(lidar_feats[0].unsqueeze(0).repeat(num_views,1,1,1), xy_bev, align_corners=False).permute(0,2,3,1) #N, H, W, C
        decorated_img_feat[~lift_mask]=0
        decorated_img_feats.append(decorated_img_feat.permute(0,3,1,2))
        decorated_img_feats = torch.stack(decorated_img_feats, dim=0)
        return decorated_img_feats

@MODELS.register_module()
class DeformableTransformer_pers(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.mask_freq = kwargs.pop("mask_freq")
        self.mask_ratio = kwargs.pop("mask_ratio")
        
        self.mask_img=kwargs.pop("mask_img", False)
        self.mask_pts=kwargs.pop("mask_pts", False)
        self.fusion_method=kwargs.get('fusion_method',False)
        self.mask_method = kwargs.get('mask_method', 'point')
        if 'patch' in self.mask_method:
            self.patch_cfg = kwargs.get('patch_cfg', None)
        if kwargs.get('residual', False):
            self.residual = kwargs.pop("residual")
        else:
            self.residual = False
        if kwargs.get('loss_weight', False):
            self.loss_weight = kwargs.pop("loss_weight")
        else:
            self.loss_weight = 1
        if self.mask_pts:
            self.model = build_deforamble_transformer(**kwargs)
        pts_channels = 256
        img_channels = 256    
        if self.mask_img:
            img_kwargs = copy.deepcopy(kwargs)
            img_kwargs['d_model'] = img_channels
            img_kwargs['num_encoder_layers'] = kwargs.get('num_img_encoder_layers',False)
            self._model = build_deforamble_transformer(**img_kwargs)
        self.conv = nn.Sequential(nn.Conv2d(
                pts_channels+img_channels, pts_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(pts_channels),
            nn.ReLU(True))
        if self.mask_pts:
            self.position_embedding = PositionEmbeddingSine(
                num_pos_feats= pts_channels // 2, normalize=True)
            self.input_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(pts_channels, pts_channels, kernel_size=1),
                        nn.GroupNorm(32, pts_channels),
                    )])        
        if self.mask_img:
            self._position_embedding = PositionEmbeddingSine(
                num_pos_feats= img_channels // 2, normalize=True)
            self._input_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(img_channels, img_channels, kernel_size=1),
                        nn.GroupNorm(32, img_channels),
                    )])
        self.num_cross_attention_layers = kwargs.get('num_cross_attention_layers', False)
        self._nheads = kwargs.get('_nheads', False)
        if self.num_cross_attention_layers or self._nheads or self.fusion_method:
            if self.mask_pts:
                self.target_proj = nn.ModuleList([
                        nn.Sequential(
                            nn.Conv2d(img_channels, img_channels, kernel_size=1),
                            nn.GroupNorm(10, img_channels),
                        )]) 
            if self.mask_img:
                self._target_proj = nn.ModuleList([
                        nn.Sequential(
                            nn.Conv2d(pts_channels, pts_channels, kernel_size=1),
                            nn.GroupNorm(32, pts_channels),
                        )]) 
        if self.mask_pts:
            self.pts_mask_tokens = nn.Parameter(torch.zeros(1, 1, pts_channels))
            self.pred = nn.Conv2d(pts_channels, pts_channels, kernel_size=1)
        if self.mask_img:
            self.img_mask_tokens = nn.Parameter(torch.zeros(1, 1, img_channels))
            self._pred = nn.Conv2d(img_channels, img_channels, kernel_size=1)
            self.linear = nn.Conv2d(img_channels, img_channels, kernel_size=1)
        if self.residual == 'concat':
            self.P_integration = ConvBNReLU(2 * pts_channels, pts_channels, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)
        self.Warp = BEVWarp()
        self.initialize_weights()
        
    def initialize_weights(self):
        if self.mask_pts:
            for proj in self.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)
        if self.mask_img:
            for __proj in self._input_proj:
                nn.init.xavier_uniform_(__proj[0].weight, gain=1)
                nn.init.constant_(__proj[0].bias, 0)
        if self.num_cross_attention_layers or self._nheads or self.fusion_method:
            if self.mask_pts:
                for _proj in self.target_proj:
                    nn.init.xavier_uniform_(_proj[0].weight, gain=1)
                    nn.init.constant_(_proj[0].bias, 0)
        if self.num_cross_attention_layers or self._nheads or self.fusion_method:
            if self.mask_img:
                for _proj_ in self._target_proj:
                    nn.init.xavier_uniform_(_proj_[0].weight, gain=1)
                    nn.init.constant_(_proj_[0].bias, 0)
        if self.mask_pts:
            torch.nn.init.normal_(self.pts_mask_tokens, std=.02)
        if self.mask_img:
            torch.nn.init.normal_(self.img_mask_tokens, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def random_patch_masking(self, _x):
        x = _x.clone()
        B, D, W, H = x.shape
        mask_tokens = self.img_mask_tokens.permute(2, 0, 1)

        mask = torch.zeros([B, W, H], device=x.device)
        for b in range(B):
            # pick size and number of patch
            l_min, l_max = self.patch_cfg.len_min, self.patch_cfg.len_max
            rand_s_l = torch.randint(l_min, l_max, (500,))
            rand_s_l_sq = torch.cumsum(torch.square(rand_s_l), dim=0)
            rand_n = (rand_s_l_sq > (W*H*self.mask_ratio)).nonzero()[0][0]

            # masking patch
            rand_x_l, rand_y_l = torch.randint(0, W, (rand_n, 1)), torch.randint(0, H, (rand_n, 1))
            for rand_s, rand_x, rand_y in zip(rand_s_l, rand_x_l, rand_y_l):
                x[b][:, rand_x:rand_x+rand_s, rand_y:rand_y+rand_s] = mask_tokens
                mask[b][rand_x:rand_x+rand_s, rand_y:rand_y+rand_s] = 1

        return x, mask.flatten(1)

    def visualize_feat(self, bev_feat, idx):
        feat = bev_feat.cpu().detach().numpy()
        min = feat.min()
        max = feat.max()
        image_features = (feat-min)/(max-min)
        image_features = (image_features*255)
        #sum_image_feature = (np.sum(np.transpose(image_features,(1,2,0)),axis=2)/64).astype("uint8")
        max_image_feature = np.max(np.transpose(image_features.astype("uint8"),(1,2,0)),axis=2)
        #sum_image_feature = cv2.applyColorMap(sum_image_feature,cv2.COLORMAP_JET)
        max_image_feature = cv2.applyColorMap(max_image_feature,cv2.COLORMAP_JET)
        #cv2.imwrite(f"max_{idx}.jpg",sum_image_feature)
        cv2.imwrite(f"max_{idx}.jpg",max_image_feature)

    def forward_single(self, img_feat, pts_feat, img_metas, pts_metas, fg_bg_mask_list, batch_index, sensor_list=None, img=None, points=None) -> torch.Tensor:
        # image feature, points feature
        if self.residual:
            residual_img = img_feat.clone()
        prob = np.random.uniform()
        if sensor_list is not None:
            cam_alive = sensor_list[0]
            pts_alive = sensor_list[1]
            _mask = (prob < self.mask_freq) * cam_alive * pts_alive
        else:
            _mask = prob < self.mask_freq
        
        ## img        
        if _mask and img_feat.requires_grad and self.mask_img:
            if self.mask_method == 'random_patch':
                img_target = img_feat.flatten(2).transpose(1, 2).clone()
                img_feat = img_feat.clone()
                _src, img_mask = self.random_patch_masking(img_feat)
                if self.residual:
                    residual_img = _src
        else:
            _src = img_feat.clone()

        if self.mask_img:
            _s_proj = self._input_proj[0](_src)
            if self.fusion_method:
                batch_size = pts_feat.shape[0]
                BN, I_C, I_H, I_W = img_feat.shape
                warped_img_feats = self.Warp(pts_feat.clone(), img_feat.view(batch_size, -1, I_C, I_H, I_W), img_metas, pts_metas, batch_index, img=img, points=points)
                B, N, C, H, W = warped_img_feats.shape
                warped_img_feats = warped_img_feats.view(B*N,C,H,W)
                # self.visualize_feat(warped_img_feats[0],'0_lidar_feat')
                _target = warped_img_feats
            else:
                _target = pts_feat.clone()
            if self.num_cross_attention_layers or self._nheads or self.fusion_method:
                _t_proj = self._target_proj[0](_target)
            else:
                _t_proj = _target
            _masks = torch.zeros(
                    (_s_proj.shape[0], _s_proj.shape[2], _s_proj.shape[3]),
                    dtype=torch.bool,
                    device=_s_proj.device,
                )
            _pos_embeds = self._position_embedding(NestedTensor(_s_proj, _masks)).to(
                    _s_proj.dtype)
            img_output = self._model([_s_proj], [_masks], [_pos_embeds], [_t_proj], query_embed=None)
            img_output = self._pred(img_output)
            img_output = img_output.contiguous()
            if self.residual == 'sum':
                img_output += residual_img
        if _mask and img_output.requires_grad and self.mask_img:
            img_output = img_output.flatten(2).transpose(1, 2)         
            loss = (img_output - img_target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
            loss = (loss * img_mask).sum() / img_mask.sum()  # mean loss on removed patches
            img_loss = self.loss_weight * loss
            img_output = img_output.transpose(1,2).view(6,256,32,88).contiguous()
        if _mask and img_output.requires_grad:
            loss_list = dict()
            if fg_bg_mask_list is not None:
                if self.mask_img:
                    loss_list['img_loss'] = img_loss
                return img_output, pts_feat, loss_list
        return img_output, pts_feat, False

    def forward(
        self, 
        img_feat, pts_feat, img_metas, pts_metas, fg_bg_mask_list, sensor_list=None, batch_input_metas=None, img=None, points=None
    ) -> torch.Tensor:
        batch = len(batch_input_metas)
        loss_list, img_feat_list, pts_feat_list = [], [], []
        loss_flag = False
        for b in range(batch):
            img_feat_ = img_feat[6*b:6*(b+1)]
            pts_feat_ = pts_feat[b:b+1]
            img_metas_ = img_metas[b]
            pts_metas_ = pts_metas
            if fg_bg_mask_list is not None:
                fg_bg_mask_list_ = [fg_bg_mask_list[0][b].unsqueeze(0),
                                    fg_bg_mask_list[1][b].unsqueeze(0)]
                sensor_list_ = [True, True]
            else:
                fg_bg_mask_list_ = None
                sensor_list_ = None
            _img_feat, _pts_feat, loss_l = self.forward_single(img_feat_, pts_feat_, img_metas_, pts_metas_, fg_bg_mask_list_, b, sensor_list_, img=img, points=points)
            img_feat_list.append(_img_feat)
            pts_feat_list.append(_pts_feat)
            loss_list.append(loss_l)
            if loss_l:
                loss_flag = True

        for b in range(batch):
            if loss_list[b]:
                if batch_input_metas[b]['smt_number'] != 2:
                    for key in list(loss_list[b].keys()):
                        loss_list[b][key] *= 0.

        if loss_flag:
            if batch == 2:
                if loss_list.count(False) == 0:
                    #  print(0)
                    for key in list(loss_list[0].keys()):
                        loss_list[0][key] += loss_list[1][key]
                        loss_list[0][key] /= 2
                else:
                    #  print(1)
                    idx = loss_list.index(False)
                    loss_list.pop(idx)
            loss_list = loss_list[0]
        else:
            #  print(2)
            loss_list = False
        return torch.stack(img_feat_list, dim=0), torch.cat(pts_feat_list, dim=0), loss_list
@MODELS.register_module()
class ModalitySpecificDecoderMask(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mask_ratio=0.5, mask_pts=False, mask_img=False, num_layers=1, kernel_size=9, bn_momentum=0.1, bias='auto', pos_emb=False):
        super(ModalitySpecificDecoderMask, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lidar_hidden_channel = 256
        self.camera_hidden_channel = 80
        self.conv = nn.Sequential(nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        # self.pts_mask_token = nn.Parameter(torch.zeros(1, self.lidar_hidden_channel, 1, 1))
        # trunc_normal_(self.pts_mask_token, mean=0., std=.02)
        self.mask_ratio = mask_ratio
        self.mask_pts = mask_pts
        self.mask_img = mask_img
        in_channels_img = in_channels[0]
        in_channels_pts = in_channels[1]
        
        decoder_embed_dim=256
        #decoder_num_heads=16
        decoder_num_heads=1
        mlp_ratio=1
        norm_layer=nn.LayerNorm
        decoder_depth=1
        self.decoder_embed_img = nn.Linear(in_channels_img, decoder_embed_dim, bias=True)
        self.decoder_embed_pts = nn.Linear(in_channels_pts, decoder_embed_dim, bias=True)
        self.img_mask_tokens = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pts_mask_tokens = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_img = nn.Parameter(torch.zeros(1, 180*180, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed_pts = nn.Parameter(torch.zeros(1, 180*180, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, 180*180 * decoder_embed_dim, bias=True) # decoder to patch
        self.initialize_weights()
        
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed_img = get_2d_sincos_pos_embed(self.decoder_pos_embed_img.shape[-1], 180, cls_token=False)
        self.decoder_pos_embed_img.data.copy_(torch.from_numpy(decoder_pos_embed_img).float().unsqueeze(0))
        decoder_pos_embed_pts = get_2d_sincos_pos_embed(self.decoder_pos_embed_pts.shape[-1], 180, cls_token=False)
        self.decoder_pos_embed_pts.data.copy_(torch.from_numpy(decoder_pos_embed_pts).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.img_mask_tokens, std=.02)
        torch.nn.init.normal_(self.pts_mask_tokens, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def img_masking(self, x):
        _x = x.clone().detach()
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)    
        
        x_masked = self.decoder_embed_img(x_masked)
        img_mask_tokens = self.img_mask_tokens.repeat(N, ids_restore.shape[1] + 1 - L, 1)
        x_ = torch.cat([x_masked, img_mask_tokens],dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # unshuffle
        return _x, x, mask
    
    def pts_masking(self, x):
        _x = x.clone().detach()
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)    
        
        x_masked = self.decoder_embed_pts(x_masked)
        pts_mask_tokens = self.pts_mask_tokens.repeat(N, ids_restore.shape[1] + 1 - L, 1)
        x_ = torch.cat([x_masked, pts_mask_tokens],dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # unshuffle
        return _x, x, mask
    
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        N,IC,H,W = inputs[0].shape
        N,PC,H,W = inputs[1].shape
        img_feat = inputs[0].view(N,IC,H*W).permute(0,2,1).contiguous()
        pts_feat = inputs[1].view(N,PC,H*W).permute(0,2,1).contiguous()
        
        prob = np.random.uniform()
        mask_img = prob < 0.5 and prob > 0.25
        mask_pts = prob < 0.25
        
        if mask_img and inputs[0].requires_grad:
            img_target, img_feat, img_mask = self.img_masking(img_feat)
        else:
            img_feat = self.decoder_embed_img(img_feat)
        
        img_feat = img_feat + self.decoder_pos_embed_img
        for blk in self.decoder_blocks:
            img_feat = blk(img_feat)
        img_feat = self.decoder_norm(img_feat)
        # predictor projection
        img_feat = self.decoder_pred(img_feat)
        
        if mask_pts and inputs[1].requires_grad:
            pts_target, pts_feat, pts_mask = self.pts_masking(pts_feat)
        else:
            pts_feat = self.decoder_embed_pts(pts_feat)
        pts_feat = pts_feat + self.decoder_pos_embed_pts
        for blk in self.decoder_blocks:
            pts_feat = blk(pts_feat)
        pts_feat = self.decoder_norm(pts_feat)
        # predictor projection
        pts_feat = self.decoder_pred(pts_feat)

        if mask_img and inputs[0].requires_grad:
            
            loss = (img_feat - img_target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

            loss = (loss * img_mask).sum() / img_mask.sum()  # mean loss on removed patches
            return self.conv(torch.cat(inputs, dim=1)), loss
        
        if mask_pts and inputs[1].requires_grad:
            loss = (pts_feat - pts_target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
            loss = (loss * pts_mask).sum() / pts_mask.sum()  # mean loss on removed patches
            return self.conv(torch.cat(inputs, dim=1)), loss
        return self.conv(torch.cat(inputs, dim=1)), False

@MODELS.register_module()
class ConvFuser(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # image feature, points feature
        return super().forward(torch.cat(inputs, dim=1))


@MODELS.register_module()
class ModalitySpecificLocalCrossAttentionlayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size):
        super(ModalitySpecificLocalCrossAttentionlayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lidar_hidden_channel = 256
        self.camera_hidden_channel = 80
        
        self.P_IML = LocalContextAttentionBlock_BEV(self.lidar_hidden_channel, self.camera_hidden_channel, self.lidar_hidden_channel, kernel_size)
        self.I_IML = LocalContextAttentionBlock_BEV(self.camera_hidden_channel, self.lidar_hidden_channel, self.camera_hidden_channel, kernel_size)
        self.P_integration = ConvBNReLU(2 * self.lidar_hidden_channel, self.lidar_hidden_channel, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)
        self.I_integration = ConvBNReLU(2 * self.camera_hidden_channel, self.camera_hidden_channel, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        img_feat = inputs[0]
        lidar_feat = inputs[1]
        I2I_feat = self.I_IML(img_feat, lidar_feat)
        new_img_feat = self.I_integration(torch.cat((I2I_feat, img_feat),dim=1))
        P2P_feat = self.P_IML(lidar_feat, img_feat)
        new_lidar_feat = self.P_integration(torch.cat((P2P_feat, lidar_feat),dim=1))
        inputs = [new_img_feat, new_lidar_feat]
        return inputs

@MODELS.register_module()
class ModalitySpecificLocalSelfAttentionlayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size):
        super(ModalitySpecificLocalSelfAttentionlayer, self).__init__()
        self.in_channels = in_channels
        self.IML = LocalContextAttentionBlock_BEV(self.in_channels, self.in_channels, self.in_channels, kernel_size)
        self.integration = ConvBNReLU(2 * self.in_channels, self.in_channels, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)

    def forward(self, inputs):
        feat = inputs
        new_feat = self.IML(feat, feat)
        new_feat = self.integration(torch.cat((new_feat, feat),dim=1))
        return new_feat

@MODELS.register_module()
class ModalitySpecificLocalCrossAttention(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_layers=1, kernel_size=9, bn_momentum=0.1,
                bias='auto'):
        super(ModalitySpecificLocalCrossAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lidar_hidden_channel = 256
        self.camera_hidden_channel = 80
        self.conv = nn.Sequential(nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.num_layers=num_layers
        self.cross_attn_list=nn.ModuleList()
        in_channels_img = in_channels[0]
        in_channels_pts = in_channels[1]
        self.shared_conv_pts = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_pts,
            in_channels_pts,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.shared_conv_img = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_img,
            in_channels_img,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        for i in range(num_layers):
            self.cross_attn_list.append(ModalitySpecificLocalCrossAttentionlayer(in_channels, out_channels, kernel_size))
        
        self.bn_momentum = bn_momentum
        self.init_weights()

    def init_weights(self):
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum
        
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        inputs[0] = self.shared_conv_img(inputs[0])
        inputs[1] = self.shared_conv_pts(inputs[1])
        for idx in range(self.num_layers):
            inputs = self.cross_attn_list[idx](inputs)
        return self.conv(torch.cat(inputs, dim=1))
def pos2embed(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb



@MODELS.register_module()
class ModalitySpecificLocalAttentionMask(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mask_ratio=0.5, mask_pts=False, mask_img=False, num_layers=1, kernel_size=9, bn_momentum=0.1, bias='auto', pos_emb=False):
        super(ModalitySpecificLocalAttentionMask, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lidar_hidden_channel = 256
        self.camera_hidden_channel = 80
        self.conv = nn.Sequential(nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        # self.pts_mask_token = nn.Parameter(torch.zeros(1, self.lidar_hidden_channel, 1, 1))
        # trunc_normal_(self.pts_mask_token, mean=0., std=.02)
        self.mask_ratio = mask_ratio
        self.mask_pts = mask_pts
        self.mask_img = mask_img
        self.num_layers=num_layers
        self.cross_attn_list=nn.ModuleList()
        self.img_self_attn_list=nn.ModuleList()
        self.pts_self_attn_list=nn.ModuleList()
        in_channels_img = in_channels[0]
        in_channels_pts = in_channels[1]
        self.shared_conv_pts = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_pts,
            in_channels_pts,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.shared_conv_img = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_img,
            in_channels_img,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        for i in range(num_layers):
            self.img_self_attn_list.append(ModalitySpecificLocalSelfAttentionlayer(in_channels_img, in_channels_img, kernel_size))
        for i in range(num_layers):
            self.pts_self_attn_list.append(ModalitySpecificLocalSelfAttentionlayer(in_channels_pts, in_channels_pts, kernel_size))
        for i in range(num_layers):
            self.cross_attn_list.append(ModalitySpecificLocalCrossAttentionlayer(in_channels, out_channels, kernel_size))
        self.pos_emb = pos_emb
        self.pts_bev_embedding = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )
        self.img_bev_embedding = nn.Sequential(
            nn.Linear(80 * 2, 80),
            nn.ReLU(inplace=True),
            nn.Linear(80, 80)
        )
        self.bn_momentum = bn_momentum
        self.init_weights()
        
    def init_weights(self):
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum
        
    def img_masking(self, img_feat):
        clean_img_feat = img_feat.clone().detach()
        BN, C, H, W = img_feat.shape
        self.img_mask_count = int(np.ceil(H*W * self.mask_ratio))
        img_mask_idx = np.random.permutation(H*W)[:self.img_mask_count]
        img_mask = np.zeros(H*W, dtype=int)
        img_mask[img_mask_idx] = 1
        img_mask = img_mask.reshape((H, W))
        img_mask = torch.tensor(img_mask).to(device=img_feat.device)
        # img_mask_tokens = self.img_mask_token.expand(BN,-1,H,W).to(device=img_feat.device)
        img_mask_tokens = torch.zeros(BN,C,H,W).to(device=img_feat.device)
        masked_img_feat = img_feat * (1-img_mask) + img_mask_tokens* img_mask
        return clean_img_feat, masked_img_feat, img_mask
    
    def pts_masking(self, pts_feat):
        clean_pts_feat = pts_feat.clone().detach()
        BN, C, H, W = pts_feat.shape
        self.pts_mask_count = int(np.ceil(H*W * self.mask_ratio))
        pts_mask_idx = np.random.permutation(H*W)[:self.pts_mask_count]
        pts_mask = np.zeros(H*W, dtype=int)
        pts_mask[pts_mask_idx] = 1
        pts_mask = pts_mask.reshape((H, W))
        pts_mask = torch.tensor(pts_mask).to(device=pts_feat.device)
        #pts_mask_tokens = self.pts_mask_token.expand(BN,-1,H,W).to(device=pts_feat.device)
        pts_mask_tokens = torch.zeros(BN,C,H,W).to(device=pts_feat.device)
        masked_pts_feat = pts_feat * (1-pts_mask) + pts_mask_tokens* pts_mask
        return clean_pts_feat, masked_pts_feat, pts_mask
    @property
    def coords_bev(self):
        grid_size = [1440,1440]
        downsample_scale = 8
        x_size, y_size = (
            grid_size[1] // downsample_scale,
            grid_size[0] // downsample_scale
        )
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = (batch_x + 0.5) / x_size
        batch_y = (batch_y + 0.5) / y_size
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        coord_base = coord_base.view(2, -1).transpose(1, 0) # (H*W, 2)
        return coord_base
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        inputs[0] = self.shared_conv_img(inputs[0])
        inputs[1] = self.shared_conv_pts(inputs[1])
        prob = np.random.uniform()

        mask_img = prob < 0.5 and prob > 0.25
        if mask_img and inputs[0].requires_grad:
            clean_img_feat, masked_img_feat, img_mask = self.img_masking(inputs[0])
            inputs[0] = masked_img_feat
        mask_pts = prob < 0.25
        if mask_pts and inputs[1].requires_grad:
            clean_pts_feat, masked_pts_feat, pts_mask = self.pts_masking(inputs[1])
            inputs[1] = masked_pts_feat
        
        if self.pos_emb:
            B,C,H,W = inputs[0].shape
            B,D,H,W = inputs[1].shape
            img_bev_pos_embeds = self.img_bev_embedding(pos2embed(self.coords_bev.to(inputs[0].device), num_pos_feats=C))
            img_bev_pos_embeds = img_bev_pos_embeds.view(H,W,-1).permute(2,0,1).repeat(B,1,1,1)
            inputs[0] = inputs[0] + img_bev_pos_embeds
            
            pts_bev_pos_embeds = self.pts_bev_embedding(pos2embed(self.coords_bev.to(inputs[1].device), num_pos_feats=D))
            pts_bev_pos_embeds = pts_bev_pos_embeds.view(H,W,-1).permute(2,0,1).repeat(B,1,1,1)
            inputs[1] = inputs[1] + pts_bev_pos_embeds

        for self_idx1 in range(self.num_layers):
            inputs[0] = self.img_self_attn_list[self_idx1](inputs[0])
        for self_idx2 in range(self.num_layers):
            inputs[1] = self.pts_self_attn_list[self_idx2](inputs[1])
        
        if mask_pts and inputs[1].requires_grad:
            pts_loss = F.l1_loss(inputs[1], clean_pts_feat, reduction='none')
            pts_loss = (pts_loss * pts_mask).sum() / (pts_mask.sum() + 1e-5) / 256
        if mask_img and inputs[0].requires_grad:
            img_loss = F.l1_loss(inputs[0], clean_img_feat, reduction='none')
            img_loss = (img_loss * img_mask).sum() / (img_mask.sum() + 1e-5) / 80
        for idx in range(self.num_layers):
            inputs = self.cross_attn_list[idx](inputs)
        if mask_pts and inputs[1].requires_grad:
            return self.conv(torch.cat(inputs, dim=1)), pts_loss
        if mask_img and inputs[0].requires_grad:
            return self.conv(torch.cat(inputs, dim=1)), img_loss
        return self.conv(torch.cat(inputs, dim=1)), False

@MODELS.register_module()
class ModalitySpecificLocalCrossAttentionMask(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mask_ratio=0.5, mask_pts=False, mask_img=False, num_layers=1, kernel_size=9, bn_momentum=0.1, bias='auto', pos_emb=False):
        super(ModalitySpecificLocalCrossAttentionMask, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lidar_hidden_channel = 256
        self.camera_hidden_channel = 80
        self.conv = nn.Sequential(nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        # self.pts_mask_token = nn.Parameter(torch.zeros(1, self.lidar_hidden_channel, 1, 1))
        # trunc_normal_(self.pts_mask_token, mean=0., std=.02)
        self.mask_ratio = mask_ratio
        self.mask_pts = mask_pts
        self.mask_img = mask_img
        self.num_layers=num_layers
        self.cross_attn_list=nn.ModuleList()
        in_channels_img = in_channels[0]
        in_channels_pts = in_channels[1]
        self.shared_conv_pts = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_pts,
            in_channels_pts,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.shared_conv_img = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_img,
            in_channels_img,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        for i in range(num_layers):
            self.cross_attn_list.append(ModalitySpecificLocalCrossAttentionlayer(in_channels, out_channels, kernel_size))
        self.pos_emb = pos_emb
        self.pts_bev_embedding = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )
        self.img_bev_embedding = nn.Sequential(
            nn.Linear(80 * 2, 80),
            nn.ReLU(inplace=True),
            nn.Linear(80, 80)
        )
        self.bn_momentum = bn_momentum
        self.init_weights()
        
    def init_weights(self):
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum
        
    def img_masking(self, img_feat):
        clean_img_feat = img_feat.clone().detach()
        BN, C, H, W = img_feat.shape
        self.img_mask_count = int(np.ceil(H*W * self.mask_ratio))
        img_mask_idx = np.random.permutation(H*W)[:self.img_mask_count]
        img_mask = np.zeros(H*W, dtype=int)
        img_mask[img_mask_idx] = 1
        img_mask = img_mask.reshape((H, W))
        img_mask = torch.tensor(img_mask).to(device=img_feat.device)
        # img_mask_tokens = self.img_mask_token.expand(BN,-1,H,W).to(device=img_feat.device)
        img_mask_tokens = torch.zeros(BN,C,H,W).to(device=img_feat.device)
        masked_img_feat = img_feat * (1-img_mask) + img_mask_tokens* img_mask
        return clean_img_feat, masked_img_feat, img_mask
    
    def pts_masking(self, pts_feat):
        clean_pts_feat = pts_feat.clone().detach()
        BN, C, H, W = pts_feat.shape
        self.pts_mask_count = int(np.ceil(H*W * self.mask_ratio))
        pts_mask_idx = np.random.permutation(H*W)[:self.pts_mask_count]
        pts_mask = np.zeros(H*W, dtype=int)
        pts_mask[pts_mask_idx] = 1
        pts_mask = pts_mask.reshape((H, W))
        pts_mask = torch.tensor(pts_mask).to(device=pts_feat.device)
        #pts_mask_tokens = self.pts_mask_token.expand(BN,-1,H,W).to(device=pts_feat.device)
        pts_mask_tokens = torch.zeros(BN,C,H,W).to(device=pts_feat.device)
        masked_pts_feat = pts_feat * (1-pts_mask) + pts_mask_tokens* pts_mask
        return clean_pts_feat, masked_pts_feat, pts_mask
    @property
    def coords_bev(self):
        grid_size = [1440,1440]
        downsample_scale = 8
        x_size, y_size = (
            grid_size[1] // downsample_scale,
            grid_size[0] // downsample_scale
        )
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = (batch_x + 0.5) / x_size
        batch_y = (batch_y + 0.5) / y_size
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        coord_base = coord_base.view(2, -1).transpose(1, 0) # (H*W, 2)
        return coord_base
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        inputs[0] = self.shared_conv_img(inputs[0])
        inputs[1] = self.shared_conv_pts(inputs[1])
        prob = np.random.uniform()

        mask_img = prob < 0.5 and prob > 0.25
        if mask_img and inputs[0].requires_grad:
            clean_img_feat, masked_img_feat, img_mask = self.img_masking(inputs[0])
            inputs[0] = masked_img_feat
        mask_pts = prob < 0.25
        if mask_pts and inputs[1].requires_grad:
            clean_pts_feat, masked_pts_feat, pts_mask = self.pts_masking(inputs[1])
            inputs[1] = masked_pts_feat
        
        if self.pos_emb:
            B,C,H,W = inputs[0].shape
            B,D,H,W = inputs[1].shape
            img_bev_pos_embeds = self.img_bev_embedding(pos2embed(self.coords_bev.to(inputs[0].device), num_pos_feats=C))
            img_bev_pos_embeds = img_bev_pos_embeds.view(H,W,-1).permute(2,0,1).repeat(B,1,1,1)
            inputs[0] = inputs[0] + img_bev_pos_embeds
            
            pts_bev_pos_embeds = self.pts_bev_embedding(pos2embed(self.coords_bev.to(inputs[1].device), num_pos_feats=D))
            pts_bev_pos_embeds = pts_bev_pos_embeds.view(H,W,-1).permute(2,0,1).repeat(B,1,1,1)
            inputs[1] = inputs[1] + pts_bev_pos_embeds

        for idx in range(self.num_layers):
            inputs = self.cross_attn_list[idx](inputs)
        if mask_pts and inputs[1].requires_grad:
            pts_loss = F.l1_loss(inputs[1], clean_pts_feat, reduction='none')
            pts_loss = (pts_loss * pts_mask).sum() / (pts_mask.sum() + 1e-5) / 256
            return self.conv(torch.cat(inputs, dim=1)), pts_loss
        if mask_img and inputs[0].requires_grad:
            img_loss = F.l1_loss(inputs[0], clean_img_feat, reduction='none')
            img_loss = (img_loss * img_mask).sum() / (img_mask.sum() + 1e-5) / 80
            return self.conv(torch.cat(inputs, dim=1)), img_loss
        return self.conv(torch.cat(inputs, dim=1)), False
    

@MODELS.register_module()
class GatedNetwork(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(GatedNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convbnrelu = nn.Sequential(nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        concat_features = sum(in_channels)
        self.conv_cf1=nn.Conv2d(in_channels=concat_features,out_channels=1, kernel_size=3, padding=1)
        self.conv_cf2=nn.Conv2d(in_channels=concat_features,out_channels=1,kernel_size=3, padding=1)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, features):
        
        img_feat = features[0]
        lidar_feat = features[1]
        Gated_features = []  
        concat_feature = torch.cat([img_feat, lidar_feat], dim=1)
        
        conv_output1 = self.conv_cf1(concat_feature)
        conv_output2 = self.conv_cf2(concat_feature)
        sigmoid_cf1 = self.sigmoid(conv_output1)
        sigmoid_cf2 = self.sigmoid(conv_output2) 
        
        img_gated_feature= sigmoid_cf1 * img_feat
        pts_gated_feature= sigmoid_cf2 * lidar_feat
        Gated_features = torch.cat([img_gated_feature, pts_gated_feature], dim=1)
        output = self.convbnrelu(Gated_features)
        return output

@MODELS.register_module()
class TransFusionHead(nn.Module):

    def __init__(
        self,
        num_proposals=128,
        auxiliary=True,
        in_channels=128 * 3,
        hidden_channel=128,
        num_classes=4,
        # config for Transformer
        num_decoder_layers=3,
        decoder_layer=dict(),
        num_heads=8,
        nms_kernel_size=1,
        bn_momentum=0.1,
        # config for FFN
        common_heads=dict(),
        num_heatmap_convs=2,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        bias='auto',
        # loss
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean'),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        # others
        train_cfg=None,
        test_cfg=None,
        bbox_coder=None,
    ):
        super(TransFusionHead, self).__init__()

        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.nms_kernel_size = nms_kernel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_heatmap = MODELS.build(loss_heatmap)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False

        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        layers = []
        layers.append(
            ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ))
        layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                hidden_channel,
                num_classes,
                kernel_size=3,
                padding=1,
                bias=bias,
            ))
        self.heatmap_head = nn.Sequential(*layers)
        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(MODELS.build(decoder_layer))

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(
                SeparateHead(
                    hidden_channel,
                    heads,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=bias,
                ))

        self.init_weights()
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training # noqa: E501
        x_size = self.test_cfg['grid_size'][0] // self.test_cfg[
            'out_size_factor']
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg[
            'out_size_factor']
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def forward_single(self, inputs, metas):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = inputs.shape[0]
        fusion_feat = self.shared_conv(inputs)

        #################################
        # image to BEV
        #################################
        fusion_feat_flatten = fusion_feat.view(batch_size,
                                               fusion_feat.shape[1],
                                               -1)  # [BS, C, H*W]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(fusion_feat.device)

        #################################
        # query initialization
        #################################
        with torch.autocast('cuda', enabled=False):
            dense_heatmap = self.heatmap_head(fusion_feat.float())
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding),
                  padding:(-padding)] = local_max_inner
        # for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg['dataset'] == 'nuScenes':
            local_max[:, 8, ] = F.max_pool2d(
                heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9, ] = F.max_pool2d(
                heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg[
                'dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
            local_max[:, 1, ] = F.max_pool2d(
                heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[:, 2, ] = F.max_pool2d(
                heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # top num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(
            dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = fusion_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, fusion_feat_flatten.shape[1], -1),
            dim=-1,
        )
        self.query_labels = top_proposals_class

        # add category embedding
        one_hot = F.one_hot(
            top_proposals_class,
            num_classes=self.num_classes).permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(
                -1, -1, bev_pos.shape[-1]),
            dim=1,
        )
        #################################
        # transformer decoder layer (Fusion feature as K,V)
        #################################
        ret_dicts = []
        for i in range(self.num_decoder_layers):
            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            query_feat = self.decoder[i](
                query_feat,
                key=fusion_feat_flatten,
                query_pos=query_pos,
                key_pos=bev_pos)

            # Prediction
            res_layer = self.prediction_heads[i](query_feat)
            res_layer['center'] = res_layer['center'] + query_pos.permute(
                0, 2, 1)
            ret_dicts.append(res_layer)

            # for next level positional embedding
            query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        ret_dicts[0]['query_heatmap_score'] = heatmap.gather(
            index=top_proposals_index[:,
                                      None, :].expand(-1, self.num_classes,
                                                      -1),
            dim=-1,
        )  # [bs, num_classes, num_proposals]
        ret_dicts[0]['dense_heatmap'] = dense_heatmap

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in [
                    'dense_heatmap', 'dense_heatmap_old', 'query_heatmap_score'
            ]:
                new_res[key] = torch.cat(
                    [ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
        return [new_res]

    def forward(self, feats, metas):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.
        Returns:
            tuple(list[dict]): Output results. first index by level, second
            index by layer
        """
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        res = multi_apply(self.forward_single, feats, [metas])
        assert len(res) == 1, 'only support one level features.'
        return res

    def predict(self, batch_feats, batch_input_metas):
        preds_dicts = self(batch_feats, batch_input_metas)
        res = self.predict_by_feat(preds_dicts, batch_input_metas)
        return res

    def predict_by_feat(self,
                        preds_dicts,
                        metas,
                        img=None,
                        rescale=False,
                        for_roi=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer
            & each batch.
        """
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_score = preds_dict[0]['heatmap'][
                ..., -self.num_proposals:].sigmoid()
            # if self.loss_iou.loss_weight != 0:
            #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid()) # noqa: E501
            one_hot = F.one_hot(
                self.query_labels,
                num_classes=self.num_classes).permute(0, 2, 1)
            batch_score = batch_score * preds_dict[0][
                'query_heatmap_score'] * one_hot

            batch_center = preds_dict[0]['center'][..., -self.num_proposals:]
            batch_height = preds_dict[0]['height'][..., -self.num_proposals:]
            batch_dim = preds_dict[0]['dim'][..., -self.num_proposals:]
            batch_rot = preds_dict[0]['rot'][..., -self.num_proposals:]
            batch_vel = None
            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][..., -self.num_proposals:]

            temp = self.bbox_coder.decode(
                batch_score,
                batch_rot,
                batch_dim,
                batch_center,
                batch_height,
                batch_vel,
                filter=True,
            )

            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(
                        num_class=8,
                        class_names=[],
                        indices=[0, 1, 2, 3, 4, 5, 6, 7],
                        radius=-1,
                    ),
                    dict(
                        num_class=1,
                        class_names=['pedestrian'],
                        indices=[8],
                        radius=0.175,
                    ),
                    dict(
                        num_class=1,
                        class_names=['traffic_cone'],
                        indices=[9],
                        radius=0.175,
                    ),
                ]
            elif self.test_cfg['dataset'] == 'Waymo':
                self.tasks = [
                    dict(
                        num_class=1,
                        class_names=['Car'],
                        indices=[0],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Pedestrian'],
                        indices=[1],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Cyclist'],
                        indices=[2],
                        radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                # adopt circle nms for different categories
                if self.test_cfg['nms_type'] is not None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat(
                                    [
                                        boxes3d[task_mask][:, :2],
                                        scores[:, None][task_mask],
                                    ],
                                    dim=1,
                                )
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                    ))
                            else:
                                boxes_for_nms = xywhr2xyxyr(
                                    metas[i]['box_type_3d'](
                                        boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_bev(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.
                                    test_cfg['post_maxsize'],
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(
                                task_mask != 0)[0][task_keep_indices]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(
                        bboxes=boxes3d[keep_mask],
                        scores=scores[keep_mask],
                        labels=labels[keep_mask],
                    )
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)

                temp_instances = InstanceData()
                temp_instances.bboxes_3d = metas[0]['box_type_3d'](
                    ret['bboxes'], box_dim=ret['bboxes'].shape[-1])
                temp_instances.scores_3d = ret['scores']
                temp_instances.labels_3d = ret['labels'].int()

                ret_layer.append(temp_instances)

            rets.append(ret_layer)
        assert len(
            rets
        ) == 1, f'only support one layer now, but get {len(rets)} layers'

        return rets[0]

    def get_targets(self, batch_gt_instances_3d: List[InstanceData],
                    preds_dict: List[dict]):
        """Generate training targets.
        Args:
            batch_gt_instances_3d (List[InstanceData]):
            preds_dict (list[dict]): The prediction results. The index of the
                list is the index of layers. The inner dict contains
                predictions of one mini-batch:
                - center: (bs, 2, num_proposals)
                - height: (bs, 1, num_proposals)
                - dim: (bs, 3, num_proposals)
                - rot: (bs, 2, num_proposals)
                - vel: (bs, 2, num_proposals)
                - cls_logit: (bs, num_classes, num_proposals)
                - query_score: (bs, num_classes, num_proposals)
                - heatmap: The original heatmap before fed into transformer
                    decoder, with shape (bs, 10, h, w)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)
                    [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        for batch_idx in range(len(batch_gt_instances_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                preds = []
                for i in range(self.num_decoder_layers):
                    pred_one_layer = preds_dict[i][key][batch_idx:batch_idx +
                                                        1]
                    preds.append(pred_one_layer)
                pred_dict[key] = torch.cat(preds)
            list_of_pred_dict.append(pred_dict)

        assert len(batch_gt_instances_3d) == len(list_of_pred_dict)
        res_tuple = multi_apply(
            self.get_targets_single,
            batch_gt_instances_3d,
            list_of_pred_dict,
            np.arange(len(batch_gt_instances_3d)),
        )
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        heatmap = torch.cat(res_tuple[7], dim=0)
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        )

    def get_targets_single(self, gt_instances_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.
        Args:
            gt_instances_3d (:obj:`InstanceData`): ground truth of instances.
            preds_dict (dict): dict of prediction result for a single sample.
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask) [1,
                    num_proposals] # noqa: E501
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
                - torch.Tensor: heatmap targets.
        """
        # 1. Assignment
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        gt_labels_3d = gt_instances_3d.labels_3d
        num_proposals = preds_dict['center'].shape[-1]

        # get pred boxes, carefully ! don't change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(
            score, rot, dim, center, height,
            vel)  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        # each layer should do label assign separately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[self.num_proposals *
                                                idx_layer:self.num_proposals *
                                                (idx_layer + 1), :]
            score_layer = score[..., self.num_proposals *
                                idx_layer:self.num_proposals *
                                (idx_layer + 1), ]

            if self.train_cfg.assigner.type == 'HungarianAssigner3D':
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    gt_labels_3d,
                    score_layer,
                    self.train_cfg,
                )
            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    None,
                    gt_labels_3d,
                    self.query_labels[batch_idx],
                )
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat(
                [res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )

        # 2. Sampling. Compatible with the interface of `PseudoSampler` in
        # mmdet.
        gt_instances, pred_instances = InstanceData(
            bboxes=gt_bboxes_tensor), InstanceData(priors=bboxes_tensor)
        sampling_result = self.bbox_sampler.sample(assign_result_ensemble,
                                                   pred_instances,
                                                   gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # 3. Create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(
            num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression
        # and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        device = labels.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]],
            dim=1).to(device)
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        feature_map_size = (grid_size[:2] // self.train_cfg['out_size_factor']
                            )  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1],
                                         feature_map_size[0])
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.train_cfg['out_size_factor']
            length = length / voxel_size[1] / self.train_cfg['out_size_factor']
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width),
                    min_overlap=self.train_cfg['gaussian_overlap'])
                radius = max(self.train_cfg['min_radius'], int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = ((x - pc_range[0]) / voxel_size[0] /
                          self.train_cfg['out_size_factor'])
                coor_y = ((y - pc_range[1]) / voxel_size[1] /
                          self.train_cfg['out_size_factor'])

                center = torch.tensor([coor_x, coor_y],
                                      dtype=torch.float32,
                                      device=device)
                center_int = center.to(torch.int32)

                # original
                # draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius) # noqa: E501
                # NOTE: fix
                draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]],
                                      center_int[[1, 0]], radius)

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (
            labels[None],
            label_weights[None],
            bbox_targets[None],
            bbox_weights[None],
            ious[None],
            int(pos_inds.shape[0]),
            float(mean_iou),
            heatmap[None],
        )

    def loss(self, batch_feats, batch_data_samples):
        """Loss function for CenterHead.

        Args:
            batch_feats (): Features in a batch.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        batch_input_metas, batch_gt_instances_3d = [], []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
        preds_dicts = self(batch_feats, batch_input_metas)
        loss = self.loss_by_feat(preds_dicts, batch_gt_instances_3d)

        return loss

    def loss_by_feat(self, preds_dicts: Tuple[List[dict]],
                     batch_gt_instances_3d: List[InstanceData], *args,
                     **kwargs):
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        ) = self.get_targets(batch_gt_instances_3d, preds_dicts[0])
        if hasattr(self, 'on_the_image_mask'):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        # compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(preds_dict['dense_heatmap']).float(),
            heatmap.float(),
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        )
        loss_dict['loss_heatmap'] = loss_heatmap

        # compute loss for each layer
        for idx_layer in range(
                self.num_decoder_layers if self.auxiliary else 1):
            if idx_layer == self.num_decoder_layers - 1 or (
                    idx_layer == 0 and self.auxiliary is False):
                prefix = 'layer_-1'
            else:
                prefix = f'layer_{idx_layer}'

            layer_labels = labels[..., idx_layer *
                                  self.num_proposals:(idx_layer + 1) *
                                  self.num_proposals, ].reshape(-1)
            layer_label_weights = label_weights[
                ..., idx_layer * self.num_proposals:(idx_layer + 1) *
                self.num_proposals, ].reshape(-1)
            layer_score = preds_dict['heatmap'][..., idx_layer *
                                                self.num_proposals:(idx_layer +
                                                                    1) *
                                                self.num_proposals, ]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(
                -1, self.num_classes)
            layer_loss_cls = self.loss_cls(
                layer_cls_score.float(),
                layer_labels,
                layer_label_weights,
                avg_factor=max(num_pos, 1),
            )

            layer_center = preds_dict['center'][..., idx_layer *
                                                self.num_proposals:(idx_layer +
                                                                    1) *
                                                self.num_proposals, ]
            layer_height = preds_dict['height'][..., idx_layer *
                                                self.num_proposals:(idx_layer +
                                                                    1) *
                                                self.num_proposals, ]
            layer_rot = preds_dict['rot'][..., idx_layer *
                                          self.num_proposals:(idx_layer + 1) *
                                          self.num_proposals, ]
            layer_dim = preds_dict['dim'][..., idx_layer *
                                          self.num_proposals:(idx_layer + 1) *
                                          self.num_proposals, ]
            preds = torch.cat(
                [layer_center, layer_height, layer_dim, layer_rot],
                dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            if 'vel' in preds_dict.keys():
                layer_vel = preds_dict['vel'][..., idx_layer *
                                              self.num_proposals:(idx_layer +
                                                                  1) *
                                              self.num_proposals, ]
                preds = torch.cat([
                    layer_center, layer_height, layer_dim, layer_rot, layer_vel
                ],
                                  dim=1).permute(
                                      0, 2,
                                      1)  # [BS, num_proposals, code_size]
            code_weights = self.train_cfg.get('code_weights', None)
            layer_bbox_weights = bbox_weights[:, idx_layer *
                                              self.num_proposals:(idx_layer +
                                                                  1) *
                                              self.num_proposals, :, ]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(  # noqa: E501
                code_weights)
            layer_bbox_targets = bbox_targets[:, idx_layer *
                                              self.num_proposals:(idx_layer +
                                                                  1) *
                                              self.num_proposals, :, ]
            layer_loss_bbox = self.loss_bbox(
                preds,
                layer_bbox_targets,
                layer_reg_weights,
                avg_factor=max(num_pos, 1))

            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox
            # loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        loss_dict['matched_ious'] = layer_loss_cls.new_tensor(matched_ious)

        return loss_dict