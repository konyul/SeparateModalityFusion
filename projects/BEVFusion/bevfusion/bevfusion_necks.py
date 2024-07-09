# modify from https://github.com/mit-han-lab/bevfusion
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from typing import List, Tuple
from mmcv.cnn.resnet import BasicBlock, make_res_layer
from mmdet3d.registry import MODELS


@MODELS.register_module()
class GeneralizedLSSFPN(BaseModule):

    def __init__(
            self,
            in_channels,
            out_channels,
            num_outs,
            start_level=0,
            end_level=-1,
            no_norm_on_lateral=False,
            conv_cfg=None,
            norm_cfg=dict(type='BN2d'),
            act_cfg=dict(type='ReLU'),
            upsample_cfg=dict(mode='bilinear', align_corners=True),
    ) -> None:
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins - 1
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i] +
                (in_channels[i + 1] if i == self.backbone_end_level -
                 1 else out_channels),
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        """Forward function."""
        # upsample -> cat -> conv1x1 -> conv3x3
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]

        # build top-down path
        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels - 1, -1, -1):
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                **self.upsample_cfg,
            )
            laterals[i] = torch.cat([laterals[i], x], dim=1)
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])

        # build outputs
        outs = [laterals[i] for i in range(used_backbone_levels)]
        return tuple(outs)

@MODELS.register_module()
class GeneralizedResNet(nn.ModuleList):
    def __init__(
        self,
        in_channels: int,
        blocks: List[Tuple[int, int, int]],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.blocks = blocks

        for num_blocks, out_channels, stride in self.blocks:
            blocks = make_res_layer(
                BasicBlock,
                in_channels,
                out_channels,
                num_blocks,
                stride=stride,
                dilation=1,
            )
            in_channels = out_channels
            self.append(blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for module in self:
            x = module(x)
            outputs.append(x)
        return outputs



@MODELS.register_module()
class LSSFPN(nn.Module):
    def __init__(
        self,
        in_indices: Tuple[int, int],
        in_channels: Tuple[int, int],
        out_channels: int,
        scale_factor: int = 1,
    ) -> None:
        super().__init__()
        self.in_indices = in_indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels[0] + in_channels[1], out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        if scale_factor > 1:
            self.upsample = nn.Sequential(
                nn.Upsample(
                    scale_factor=scale_factor,
                    mode="bilinear",
                    align_corners=True,
                ),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x1 = x[self.in_indices[0]]
        assert x1.shape[1] == self.in_channels[0]

        x2 = x[self.in_indices[1]]
        assert x2.shape[1] == self.in_channels[1]

        x1 = F.interpolate(
            x1,
            size=x2.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        x = torch.cat([x1, x2], dim=1)

        x = self.fuse(x)
        if self.scale_factor > 1:
            x = self.upsample(x)
        return x