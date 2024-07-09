# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models import DetrTransformerDecoderLayer
from torch import Tensor, nn

from mmdet3d.registry import MODELS
import torch
from mmcv.cnn.bricks.transformer import MultiheadAttention
import torch.utils.checkpoint as cp
import copy
class PositionEncodingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats), nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


@MODELS.register_module()
class TransformerDecoderLayer(DetrTransformerDecoderLayer):

    def __init__(self,
                 pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128),
                 with_cp=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.self_posembed = PositionEncodingLearned(**pos_encoding_cfg)
        self.cross_posembed = PositionEncodingLearned(**pos_encoding_cfg)
        self.with_cp = with_cp

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                **kwargs) -> Tensor:
        # """
        # Args:
        #     query (Tensor): The input query, has shape (bs, num_queries, dim).
        #     key (Tensor, optional): The input key, has shape (bs, num_keys,
        #         dim). If `None`, the `query` will be used. Defaults to `None`.
        #     value (Tensor, optional): The input value, has the same shape as
        #         `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
        #         `key` will be used. Defaults to `None`.
        #     query_pos (Tensor, optional): The positional encoding for `query`,
        #         has the same shape as `query`. If not `None`, it will be added
        #         to `query` before forward function. Defaults to `None`.
        #     key_pos (Tensor, optional): The positional encoding for `key`, has
        #         the same shape as `key`. If not `None`, it will be added to
        #         `key` before forward function. If None, and `query_pos` has the
        #         same shape as `key`, then `query_pos` will be used for
        #         `key_pos`. Defaults to None.
        #     self_attn_mask (Tensor, optional): ByteTensor mask, has shape
        #         (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
        #         Defaults to None.
        #     cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
        #         (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
        #         Defaults to None.
        #     key_padding_mask (Tensor, optional): The `key_padding_mask` of
        #         `self_attn` input. ByteTensor, has shape (bs, num_value).
        #         Defaults to None.

        # Returns:
        #     Tensor: forwarded results, has shape (bs, num_queries, dim).
        # """
        if self.self_posembed is not None and query_pos is not None:
            query_pos = self.self_posembed(query_pos).transpose(1, 2)
        else:
            query_pos = None
        if self.cross_posembed is not None and key_pos is not None:
            key_pos = self.cross_posembed(key_pos).transpose(1, 2)
        else:
            key_pos = None
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        # Note that the `value` (equal to `query`) is encoded with `query_pos`.
        # This is different from the standard DETR Decoder Layer.
        if self.with_cp:
            query = cp.checkpoint(self.self_attn, query, query, query + query_pos, None, query_pos, query_pos, self_attn_mask)
        else: 
            query = self.self_attn(
                query=query,
                key=query,
                value=query + query_pos,
                query_pos=query_pos,
                key_pos=query_pos,
                attn_mask=self_attn_mask,
                **kwargs)
        query = self.norms[0](query)
        # Note that the `value` (equal to `key`) is encoded with `key_pos`.
        # This is different from the standard DETR Decoder Layer.
        if self.with_cp:
            query = cp.checkpoint(self.cross_attn, query, key, key + key_pos, None, query_pos, key_pos, cross_attn_mask, key_padding_mask)
        else:
            query = self.cross_attn(
                query=query,
                key=key,
                value=key + key_pos,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_mask=cross_attn_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        query = query.transpose(1, 2)
        return query

@MODELS.register_module()
class CMTransformerDecoderLayer(DetrTransformerDecoderLayer):

    def __init__(self,
                 pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128),
                 hybrid_query=False,
                 multi_value=False,
                 with_cp=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.hybrid_query = hybrid_query
        self.multi_value = multi_value
        if self.hybrid_query:
            self.self_posembed = PositionEncodingLearned(**pos_encoding_cfg)
            self.self_posembedv2 = PositionEncodingLearned(**pos_encoding_cfg)
            self.self_posembedv3 = PositionEncodingLearned(**pos_encoding_cfg)
        else:
            self.self_posembed = PositionEncodingLearned(**pos_encoding_cfg)
        self.cross_posembed = PositionEncodingLearned(**pos_encoding_cfg)
        self.cross_posembedv2 = PositionEncodingLearned(**pos_encoding_cfg)
        self.cross_posembedv3 = PositionEncodingLearned(**pos_encoding_cfg)
        self.cross_attn_img = MultiheadAttention(**self.cross_attn_cfg)
        self.cross_attn_pts = MultiheadAttention(**self.cross_attn_cfg)
        self.with_cp = with_cp
        self.i_query_norm = copy.deepcopy(self.norms[1])
        self.p_query_norm = copy.deepcopy(self.norms[1])

    def forward(self,
                query,
                key = None,
                value: Tensor = None,
                query_pos = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                **kwargs) -> Tensor:
        # """
        # Args:
        #     query (Tensor): The input query, has shape (bs, num_queries, dim).
        #     key (Tensor, optional): The input key, has shape (bs, num_keys,
        #         dim). If `None`, the `query` will be used. Defaults to `None`.
        #     value (Tensor, optional): The input value, has the same shape as
        #         `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
        #         `key` will be used. Defaults to `None`.
        #     query_pos (Tensor, optional): The positional encoding for `query`,
        #         has the same shape as `query`. If not `None`, it will be added
        #         to `query` before forward function. Defaults to `None`.
        #     key_pos (Tensor, optional): The positional encoding for `key`, has
        #         the same shape as `key`. If not `None`, it will be added to
        #         `key` before forward function. If None, and `query_pos` has the
        #         same shape as `key`, then `query_pos` will be used for
        #         `key_pos`. Defaults to None.
        #     self_attn_mask (Tensor, optional): ByteTensor mask, has shape
        #         (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
        #         Defaults to None.
        #     cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
        #         (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
        #         Defaults to None.
        #     key_padding_mask (Tensor, optional): The `key_padding_mask` of
        #         `self_attn` input. ByteTensor, has shape (bs, num_value).
        #         Defaults to None.

        # Returns:
        #     Tensor: forwarded results, has shape (bs, num_queries, dim).
        # """
        if self.self_posembed is not None and query_pos is not None:
            if self.hybrid_query:
                f_query_pos = self.self_posembed(query_pos[0]).transpose(1, 2)
                i_query_pos = self.self_posembedv2(query_pos[1]).transpose(1, 2)
                p_query_pos = self.self_posembedv3(query_pos[2]).transpose(1, 2)
            else:
                f_query_pos = self.self_posembed(query_pos).transpose(1, 2)
        else:
            query_pos = None
        if self.cross_posembed is not None and key_pos is not None:
            f_key_pos = self.cross_posembed(key_pos).transpose(1, 2)
            i_key_pos = self.cross_posembedv2(key_pos).transpose(1, 2)
            p_key_pos = self.cross_posembedv3(key_pos).transpose(1, 2)
        else:
            key_pos = None
        if self.hybrid_query:
            query = [_query.transpose(1, 2) for _query in query]
        else:
            query = query.transpose(1, 2)
        key = [_key.transpose(1, 2) for _key in key]
        # Note that the `value` (equal to `query`) is encoded with `query_pos`.
        # This is different from the standard DETR Decoder Layer.
        if self.with_cp:
            if self.hybrid_query:
                query = cp.checkpoint(self.self_attn, torch.cat(query, dim=1),torch.cat(query, dim=1),torch.cat(query, dim=1) + torch.cat([f_query_pos, i_query_pos, p_query_pos],dim=1), None, torch.cat([f_query_pos, i_query_pos, p_query_pos],dim=1),torch.cat([f_query_pos, i_query_pos, p_query_pos],dim=1),self_attn_mask)
            else:
                query = cp.checkpoint(self.self_attn, query, query, query+f_query_pos, None, f_query_pos, f_query_pos, self_attn_mask)
        else:
            if self.hybrid_query:
                query = self.self_attn(
                    query=torch.cat(query, dim=1),
                    key=torch.cat(query, dim=1),
                    value=torch.cat(query, dim=1) + torch.cat([f_query_pos, i_query_pos, p_query_pos],dim=1),
                    query_pos=torch.cat([f_query_pos, i_query_pos, p_query_pos],dim=1),
                    key_pos=torch.cat([f_query_pos, i_query_pos, p_query_pos],dim=1),
                    attn_mask=self_attn_mask,
                    **kwargs)
            else:
                query = self.self_attn(
                    query=query,
                    key=query,
                    value=query + f_query_pos,
                    query_pos=f_query_pos,
                    key_pos=f_query_pos,
                    attn_mask=self_attn_mask,
                    **kwargs)
        query = self.norms[0](query)
        f_num_proposal = f_query_pos.shape[1]
        # Note that the `value` (equal to `key`) is encoded with `key_pos`.
        # This is different from the standard DETR Decoder Layer.
        if self.with_cp:
            if self.hybrid_query:
                f_query = cp.checkpoint(self.cross_attn, query[:,:f_num_proposal,:], key[0], key[0] + f_key_pos, None, f_query_pos, f_key_pos, cross_attn_mask,key_padding_mask)
                i_query = cp.checkpoint(self.cross_attn_img, query[:,f_num_proposal:f_num_proposal+100,:], key[1], key[1] + i_key_pos, None, i_query_pos, i_key_pos, cross_attn_mask,key_padding_mask)
                p_query = cp.checkpoint(self.cross_attn_pts, query[:,f_num_proposal+100:f_num_proposal+200,:], key[2], key[2] + p_key_pos, None, p_query_pos, p_key_pos, cross_attn_mask,key_padding_mask)
            else:
                f_query = cp.checkpoint(self.cross_attn, query, key[0], key[0]+f_key_pos, None, f_query_pos, f_key_pos, cross_attn_mask, key_padding_mask)
                i_query = cp.checkpoint(self.cross_attn_img, query, key[1], key[1] + i_key_pos, None, f_query_pos, i_key_pos, cross_attn_mask, key_padding_mask)
                p_query = cp.checkpoint(self.cross_attn_pts, query, key[2], key[2] + p_key_pos, None, f_query_pos, p_key_pos, cross_attn_mask, key_padding_mask)
        else:
            if self.hybrid_query:
                f_query = self.cross_attn(
                    query=query[:,:f_num_proposal,:],
                    key=key[0],
                    value=key[0] + f_key_pos,
                    query_pos=f_query_pos,
                    key_pos=f_key_pos,
                    attn_mask=cross_attn_mask,
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                i_query = self.cross_attn_img(
                    query=query[:,f_num_proposal:f_num_proposal+100,:],
                    key=key[1],
                    value=key[1] + i_key_pos,
                    query_pos=i_query_pos,
                    key_pos=i_key_pos,
                    attn_mask=cross_attn_mask,
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                p_query = self.cross_attn_pts(
                    query=query[:,f_num_proposal+100:f_num_proposal+200,:],
                    key=key[2],
                    value=key[2] + p_key_pos,
                    query_pos=p_query_pos,
                    key_pos=p_key_pos,
                    attn_mask=cross_attn_mask,
                    key_padding_mask=key_padding_mask,
                    **kwargs)
            else:
                f_query = self.cross_attn(
                    query=query,
                    key=key[0],
                    value=key[0] + f_key_pos,
                    query_pos=f_query_pos,
                    key_pos=f_key_pos,
                    attn_mask=cross_attn_mask,
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                i_query = self.cross_attn_img(
                    query=query,
                    key=key[1],
                    value=key[1] + i_key_pos,
                    query_pos=f_query_pos,
                    key_pos=i_key_pos,
                    attn_mask=cross_attn_mask,
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                p_query = self.cross_attn_pts(
                    query=query,
                    key=key[2],
                    value=key[2] + p_key_pos,
                    query_pos=f_query_pos,
                    key_pos=p_key_pos,
                    attn_mask=cross_attn_mask,
                    key_padding_mask=key_padding_mask,
                    **kwargs)
        f_query = self.norms[1](f_query)
        i_query = self.i_query_norm(i_query)
        p_query = self.p_query_norm(p_query)
        if self.hybrid_query:
            query = torch.cat([f_query, i_query, p_query], dim=1).contiguous()
        if self.multi_value == 'sum':
            query = torch.sum(torch.stack([f_query, i_query, p_query]), dim=0)
        elif self.multi_value == 'max':
            query = torch.max(torch.stack([f_query, i_query, p_query]), dim=0)[0]
        if not self.hybrid_query and self.multi_value not in ['sum', 'max']:
            AssertionError('Queries are not fused')
        query = self.ffn(query)
        query = self.norms[2](query)

        query = query.transpose(1, 2)
        return query