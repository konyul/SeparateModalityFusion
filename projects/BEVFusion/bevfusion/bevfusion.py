from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization
from mmdet3d.structures.ops import box_np_ops
import cv2
@MODELS.register_module()
class BEVFusion(Base3DDetector):

    def __init__(
        self,
        freeze_img=False,
        freeze_pts=False,
        sep_fg=False,
        smt=False,
        use_pts_feat=False,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        imgpts_neck: Optional[dict] = None,
        masking_encoder: Optional[dict] = None,
        img_backbone_decoder: Optional[dict] = None,
        img_neck_decoder: Optional[dict] = None,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        if 'pillarize_cfg' in data_preprocessor:
            pillarize_cfg = data_preprocessor.pop('pillarize_cfg')
        else:
            pillarize_cfg = False
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)
        if pillarize_cfg:
            self.pts_pillar_layer = Voxelization(**pillarize_cfg)
        else:
            self.pts_pillar_layer = False

        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None

        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)
        self.imgpts_neck = MODELS.build(imgpts_neck) if imgpts_neck else None
        self.masking_encoder = MODELS.build(masking_encoder) if masking_encoder else None
        self.head_name = bbox_head['type']
        self.bbox_head = MODELS.build(bbox_head)
        self.img_backbone_decoder = MODELS.build(img_backbone_decoder) if img_backbone_decoder else None
        self.img_neck_decoder = MODELS.build(img_neck_decoder) if img_neck_decoder else None
        self.use_pts_feat = use_pts_feat
        self.freeze_img = freeze_img
        self.freeze_pts = freeze_pts
        self.sep_fg = sep_fg
        self.smt = smt
        self.init_weights()

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()
        if self.freeze_img:
            if self.img_backbone is not None:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False

        if self.freeze_pts:
            if self.pts_voxel_encoder is not None:
                for param in self.pts_voxel_encoder.parameters():
                    param.requires_grad = False
            if self.pts_middle_encoder is not None:
                for param in self.pts_middle_encoder.parameters():
                    param.requires_grad = False
            if self.fusion_layer is not None:
                for param in self.fusion_layer.parameters():
                    param.requires_grad = False
            # for name, param in self.named_parameters():
            #     if 'pts' in name and 'pts_bbox_head' not in name and 'imgpts_neck' not in name:
            #         param.requires_grad = False
            #     if 'pts_bbox_head.decoder.0' in name:
            #         param.requires_grad = False
            #     if 'imgpts_neck.shared_conv_pts' in name:
            #         param.requires_grad = False
            #     if 'pts_bbox_head.heatmap_head' in name and 'pts_bbox_head.heatmap_head_img' not in name:
            #         param.requires_grad = False
            #     if 'pts_bbox_head.prediction_heads.0' in name:
            #         param.requires_grad = False
            #     if 'pts_bbox_head.class_encoding' in name:
            #         param.requires_grad = False
            # def fix_bn(m):
            #     if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            #         m.track_running_stats = False
            # self.pts_voxel_layer.apply(fix_bn)
            # self.pts_voxel_encoder.apply(fix_bn)
            # self.pts_middle_encoder.apply(fix_bn)
            # self.pts_backbone.apply(fix_bn)
            # self.pts_neck.apply(fix_bn)
            # self.pts_bbox_head.heatmap_head.apply(fix_bn)
            # self.pts_bbox_head.class_encoding.apply(fix_bn)
            # self.pts_bbox_head.decoder[0].apply(fix_bn)
            # self.pts_bbox_head.prediction_heads[0].apply(fix_bn)            
            # self.imgpts_neck.shared_conv_pts.apply(fix_bn)
        
        
    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None
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
    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        pts_feats,
        pts_metas,
        fg_bg_mask_list=None,
        sensor_list=None,
        batch_input_metas=None
    ) -> torch.Tensor:

        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()
        # x_ = x.clone()
        x = self.img_backbone(x)
        x = self.img_neck(x)
        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        if self.imgpts_neck is not None:
            x, pts_feats, mask_loss = self.imgpts_neck(x, pts_feats, img_metas, pts_metas, fg_bg_mask_list, sensor_list, batch_input_metas)#, img=x_, points=points)
            x = x.contiguous()
        x = x.view(B, int(BN / B), C, H, W)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        if self.imgpts_neck is not None:
            return x, pts_feats, mask_loss
        else:
            return x, pts_feats, None

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            if self.pts_pillar_layer:
                pts_metas = self.voxelize(points, voxel_type='pillar')
            else:
                pts_metas = None
            batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x, pts_metas

    @torch.no_grad()
    def voxelize(self, points, voxel_type='voxel'):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            if voxel_type == 'voxel':
                ret = self.pts_voxel_layer(res)
            elif voxel_type == 'pillar':
                ret = self.pts_pillar_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n) # num_points
        feats = torch.cat(feats, dim=0) # voxels
        coords = torch.cat(coords, dim=0) # coors
        
        if voxel_type == 'pillar':
            pts_metas = {}
            pts_metas['pillars'] = feats
            pts_metas['pillar_coors'] = coords
            pts_metas['pts'] = points
        
        # HardSimpleVFE
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()
        if voxel_type == 'pillar':
            pts_metas['pillar_center'] = feats
            pts_metas['pillars_num_points'] = sizes
            return pts_metas
        return feats, coords, sizes

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats, _, _, cm_feat = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            if self.head_name == "RobustHead":
                outputs = self.bbox_head.predict(feats, cm_feat, batch_input_metas)
            else:
                outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        fg_bg_mask_list=None,
        sensor_list=None,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        pts_feature, pts_metas = self.extract_pts_feat(batch_inputs_dict)
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature, pts_feature, mask_loss = self.extract_img_feat(imgs, deepcopy(points),
                                                lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix,
                                                lidar_aug_matrix,
                                                batch_input_metas,
                                                pts_feature,
                                                pts_metas,
                                                fg_bg_mask_list,
                                                sensor_list,
                                                batch_input_metas)
            features.append(img_feature)
        features.append(pts_feature)
        if self.fusion_layer is not None:
            if 'mask_ratio' in self.fusion_layer.__dict__:
                x, pts_loss = self.fusion_layer(features, fg_bg_mask_list, sensor_list, batch_input_metas)
            else:
                x = self.fusion_layer(features)
                pts_loss = None
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        if self.img_backbone_decoder is not None:
            img_feature = self.img_backbone_decoder(img_feature.clone())
        if self.img_neck_decoder is not None:   
            img_feature = self.img_neck_decoder(img_feature)
        if self.use_pts_feat:
            pts_feature = self.pts_backbone(pts_feature.clone())
            pts_feature = self.pts_neck(pts_feature)
            pts_feature = pts_feature[0]
        return x, mask_loss, pts_loss, [img_feature, pts_feature]

    def fg_bg_mask(self, batch_data_samples):
            
        grid_size = torch.tensor([1440, 1440, 1])
        pc_range = torch.tensor([-54.0, -54.0, -5.0, 54.0, 54.0, 3.0])
        voxel_size = torch.tensor([0.075, 0.075, 0.2])

        B, H, W = len(batch_data_samples), 180, 180

        assert grid_size[0] == grid_size[1] and W == H
        assert grid_size[0] % W == 0
        out_size_factor = torch.div(grid_size[0], W, rounding_mode='floor')

        coord_xs = [i * voxel_size[0] * out_size_factor + pc_range[0] for i in range(W)]
        coord_ys = [i * voxel_size[1] * out_size_factor + pc_range[1] for i in range(H)]
        coord_xs, coord_ys = np.meshgrid(coord_xs, coord_ys, indexing='ij')
        coord_xs = coord_xs.reshape(-1, 1)
        coord_ys = coord_ys.reshape(-1, 1)

        coord_zs = np.ones_like(coord_xs) * 0.5
        coords = np.hstack((coord_xs, coord_ys, coord_zs))
        assert coords.shape[0] == W * W and coords.shape[1] == 3
        
        device = torch.device('cpu')
        coords = torch.as_tensor(coords, dtype=torch.float32, device=device)
        
        fg_masks = []
        bg_masks = []
        
        for sample in batch_data_samples:
            boxes = sample.gt_instances_3d['bboxes_3d']
            points = coords.numpy()
            boxes = deepcopy(boxes.detach().cpu().numpy())
            boxes[:, 2] = 0
            boxes[:, 5] = 1
            mask = box_np_ops.points_in_rbbox(points, boxes)

            fg_mask = mask.any(axis=-1).astype(float)
            bg_mask = abs(fg_mask-1)

            fg_mask = fg_mask.reshape(1, 1, H, W)
            bg_mask = bg_mask.reshape(1, 1, H, W)
            fg_masks.append(torch.tensor(fg_mask))
            bg_masks.append(torch.tensor(bg_mask))

        fg_mask = torch.cat(fg_masks, dim=0).float()
        bg_mask = torch.cat(bg_masks, dim=0).float()

        return [fg_mask, bg_mask]

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        if self.sep_fg:
            fg_bg_mask_list = self.fg_bg_mask(batch_data_samples)
        else:
            fg_bg_mask_list = None
        if 'sensor_list' in batch_inputs_dict:
            sensor_list = batch_inputs_dict['sensor_list']
        else:
            sensor_list = None
        feats, mask_loss, pts_loss, cm_feat = self.extract_feat(batch_inputs_dict, batch_input_metas, fg_bg_mask_list, sensor_list)
        losses = dict()
        if self.with_bbox_head:
            if self.head_name == 'RobustHead':
                bbox_loss = self.bbox_head.loss(feats, cm_feat, batch_data_samples)
            else:
                bbox_loss = self.bbox_head.loss(feats, batch_data_samples)
        if pts_loss:
            if isinstance(pts_loss,dict):
                losses.update(pts_loss) 
        if mask_loss:
            if isinstance(mask_loss,dict):
                losses.update(mask_loss) 
        losses.update(bbox_loss)
        return losses