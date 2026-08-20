# Copyright (c) OpenMMLab. All rights reserved.
"""带尺度、通道和空间注意力的SegFormer解码头。"""

import torch
import torch.nn as nn

from mmseg.models.decode_heads.segformer_head import SegformerHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS


class ScaleChannelAttention(nn.Module):
    """为每个尺度、每个通道生成自适应融合权重。

    最后一层使用零初始化。经过softmax并乘以尺度数量后，初始权重全部为1，
    因而训练开始时与原始SegFormerHead的特征幅值保持一致。
    """

    def __init__(self, channels, num_scales, reduction=16):
        super().__init__()
        hidden_channels = max(channels // reduction, 16)
        self.num_scales = int(num_scales)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
        self.expand = nn.Conv2d(
            hidden_channels,
            channels * self.num_scales,
            kernel_size=1,
            bias=True)

        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, features):
        summed_feature = features[0]
        for feature in features[1:]:
            summed_feature = summed_feature + feature
        descriptor = self.pool(summed_feature)
        logits = self.expand(self.reduce(descriptor))
        batch_size, _, _, _ = logits.shape
        logits = logits.view(
            batch_size, self.num_scales, -1, 1, 1)
        weights = torch.softmax(logits, dim=1) * self.num_scales
        return [
            feature * weights[:, index]
            for index, feature in enumerate(features)
        ]


class ChannelAttention(nn.Module):
    """CBAM风格的通道注意力。"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_channels = max(channels // reduction, 16)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        attention = (
            self.shared(self.avg_pool(feature)) +
            self.shared(self.max_pool(feature)))
        return self.sigmoid(attention)


class SpatialAttention(nn.Module):
    """CBAM风格的空间注意力。"""

    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError('spatial_kernel_size只能为3或7')
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        average = torch.mean(feature, dim=1, keepdim=True)
        maximum, _ = torch.max(feature, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([average, maximum], dim=1)))


@MODELS.register_module()
class AttentionSegformerHead(SegformerHead):
    """SegFormerHead加多尺度、通道和空间注意力。

    通道和空间注意力采用带可学习系数的残差形式。两个系数初始化为0，
    因此模型起始状态与原始SegFormerHead一致，之后由训练自动学习注意力强度。
    """

    def __init__(self,
                 attention_reduction=16,
                 spatial_kernel_size=7,
                 **kwargs):
        super().__init__(**kwargs)
        num_scales = len(self.in_channels)
        self.scale_attention = ScaleChannelAttention(
            channels=self.channels,
            num_scales=num_scales,
            reduction=attention_reduction)
        self.channel_attention = ChannelAttention(
            channels=self.channels,
            reduction=attention_reduction)
        self.spatial_attention = SpatialAttention(
            kernel_size=spatial_kernel_size)

        # 恒等初始化，避免新注意力模块破坏原始稳定训练状态。
        self.channel_scale = nn.Parameter(torch.zeros(1))
        self.spatial_scale = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        # 四个主干特征的分辨率分别约为输入的1/4、1/8、1/16和1/32。
        inputs = self._transform_inputs(inputs)
        projected_features = []

        for index, feature in enumerate(inputs):
            projected = self.convs[index](feature)
            projected = resize(
                input=projected,
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners)
            projected_features.append(projected)

        attended_features = self.scale_attention(projected_features)
        fused = self.fusion_conv(torch.cat(attended_features, dim=1))

        channel_weight = self.channel_attention(fused)
        fused = fused * (1.0 + self.channel_scale * channel_weight)

        spatial_weight = self.spatial_attention(fused)
        fused = fused * (1.0 + self.spatial_scale * spatial_weight)

        return self.cls_seg(fused)
