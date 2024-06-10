import copy
import warnings
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
#from mmcv.runner import force_fp32, auto_fp16
from .fp16_utils import force_fp32, auto_fp16
import numpy as np
import torch
import cv2 as cv
import mmcv
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils import digit_version
from mmengine.model import BaseModule, ModuleList, Sequential
from mmcv.cnn import (Linear, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmengine.registry import MODELS
from .spatial_cross_attention_pts import SpatialCrossAttentionPts
import cv2

class PtsEncoder(BaseModule):

    """
    Attention with both self and cross
    Implements the encoder in BEVFormer transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args,
                 pc_range=None,
                 num_points_in_pillar_lidar=1,
                 return_intermediate=False,
                 dataset_type='nuscenes',
                 **kwargs):
        super().__init__()
        if isinstance(kwargs['transformerlayers'], dict):
            transformerlayers = [
                copy.deepcopy(kwargs['transformerlayers']) for _ in range(kwargs['num_layers'])
            ]
        else:
            assert isinstance(kwargs['transformerlayers'], list) and \
                   len(kwargs['transformerlayers']) == kwargs['num_layers']
        self.num_layers = kwargs['num_layers']
        self.layers = ModuleList()
        for i in range(kwargs['num_layers']):
            self.layers.append(PtsLayer(**transformerlayers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm
        
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar_lidar = num_points_in_pillar_lidar

        self.pc_range = pc_range
        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H

            # ## debug
            # print('\nVariables in get_reference_points:'
            #       '\nzs size:', zs.size(),
            #       '\nxs size:', xs.size(),
            #       '\nys size:', ys.size())
            # ##

            ref_3d = torch.stack((xs, ys, zs), -1)
            # ## debug
            # print('ref 3d stack xs ys zs:',ref_3d.size())
            # ##
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            # ## debug
            # print('ref 3d permute flatten and permute:',ref_3d.size())
            # ##
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            # ## debug
            # print('ref 3d expand:',ref_3d.size()) # ref_3d: [2, 4, 40000, 3]
            # ##

            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2) # ref_2d: [2, 40000, 1, 2]
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points):

        reference_points = reference_points.clone()# [2, 4, 40000, 3]

        reference_points = reference_points.permute(1, 0, 2, 3) # [4, 2, 40000, 3]
        reference_points_lidar = reference_points[...,:2] #[4, 2, 40000,2]

        bev_mask = ((reference_points_lidar[..., 1:2] > 0.0)
                    & (reference_points_lidar[..., 1:2] < 1.0)
                    & (reference_points_lidar[..., 0:1] < 1.0)
                    & (reference_points_lidar[..., 0:1] > 0.0))

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))
        # (4, 2, 40000, 1)
        bev_mask = bev_mask.permute(1,2,0,3).squeeze(-1)
        # [2, 40000, 4, 1] -> [2, 40000, 4]

        return reference_points_lidar, bev_mask
    def visualize_feat(self, bev_feat, idx):
        feat = bev_feat.cpu().detach().numpy()
        min = feat.min()
        max = feat.max()
        image_features = (feat-min)/(max-min)
        image_features = (image_features*255)
        max_image_feature = np.max(np.transpose(image_features.astype("uint8"),(1,2,0)),axis=2)
        max_image_feature = cv2.applyColorMap(max_image_feature,cv2.COLORMAP_JET)
        cv2.imwrite(f"max_{idx}.jpg",max_image_feature)
        
    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                cross_modal_feat=None,
                **kwargs):
        """Forward function for `UniBEVEncoder`.

        """

        output = bev_query
        intermediate = []
        ref_3d_lidar = self.get_reference_points(
            bev_h,
            bev_w,
            self.pc_range[5]-self.pc_range[2],
            self.num_points_in_pillar_lidar,
            dim='3d',
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(
            bev_h,
            bev_w,
            dim='2d',
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype)
        reference_points_lidar, _ = self.point_sampling(ref_3d_lidar)

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)

        ## new_properties_shiming
        if bev_pos is not None:
            bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape # [2, 40000, 1, 2]
        # ## debug:
        # print('bev_queries shape:', bev_query.size())
        # ##

        # Bev queries in encoder: torch.Size([2, 40000, 256]) (bs, bev_h x bev_w, embed_dims)
        # Keys in encoder (feat flatten): torch.Size([6, 920, 2, 256]) (num_cam, h_img_feats x w_img_feats, bs, embed_dims)
        # Value in encoder (feat flatten): torch.Size([6, 920, 2, 256]) (num_cam, h_img_feats x w_img_feats, bs, embed_dims)

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=ref_2d,
                ref_3d=ref_3d_lidar,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_lidar=reference_points_lidar,
                cross_modal_feat=cross_modal_feat,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


def build_attention(cfg, default_args=None):
    """Builder for attention."""
    return MODELS.build(cfg, default_args=default_args)
def build_feedforward_network(cfg, default_args=None):
    """Builder for feed-forward network (FFN)."""
    return MODELS.build(cfg, default_args=default_args)

class PtsLayer(BaseModule):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 batch_first = True,
                 ffn_num_fcs=2,
                 **kwargs):
        super().__init__()
        
        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=kwargs['ffn_cfgs']['embed_dims'],
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ', DeprecationWarning)
                ffn_cfgs[new_name] = kwargs[ori_name]
        self.batch_first = batch_first

        assert set(operation_order) & {
            'self_attn', 'norm', 'ffn', 'cross_attn'} == \
            set(operation_order), f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all four operation type ' \
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                if operation_name == 'cross_attn':
                    attention = SpatialCrossAttentionPts(**attn_cfgs[index])
                else:
                    attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index],
                                          dict(type='FFN')))

        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])
            
        self.fp16_enabled = False

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_lidar=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                cross_modal_feat=None,
                # prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.
        """

        # ## debug
        # print('In BEVVoxelDETRLayer:')
        # # print('kwargs keys:', kwargs.keys())
        # # print('bev mask:', kwargs['bev_mask'].size())
        # print('reference_point_lidar:', reference_points_lidar.size())
        # ##

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    query,
                    query,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query
                # ## debug
                # print('output query:', query.size())
                # ## debug
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos, # todo: query_pos
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_lidar=reference_points_lidar,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    cross_modal_feat=cross_modal_feat,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

