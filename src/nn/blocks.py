"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

Conv building blocks shared by the necks: the re-parameterisable conv-BN layers and the
CSP / GELAN fusion blocks the hybrid encoder's feature pyramid is made of. Every block that
folds its BatchNorm (or its parallel branches) into a single conv at inference does so in
``convert_to_deploy``, which ``DOME.deploy`` calls on every module that has one.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .backbone.common import ConvNormLayer, get_activation

__all__ = [
    "CSPLayer",
    "ChannelAttention",
    "ConvNormLayerFuse",
    "DepthwiseSeparableConv",
    "RepNCSPELAN4",
    "SCDown",
    "VGGBlock",
    "fuse_conv_bn",
]


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel and bias of the single conv equivalent to ``conv`` followed by ``bn``."""
    std = (bn.running_var + bn.eps).sqrt()
    scale = bn.weight / std
    return conv.weight * scale.reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * scale


class ConvNormLayerFuse(nn.Module):
    """Conv -> BN -> act, foldable into a single conv for deployment."""

    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = (
            ch_in,
            ch_out,
            kernel_size,
            stride,
            g,
            padding,
            bias,
        )

    def forward(self, x):
        if hasattr(self, "conv_bn_fused"):
            return self.act(self.conv_bn_fused(x))
        return self.act(self.norm(self.conv(x)))

    def convert_to_deploy(self):
        if not hasattr(self, "conv_bn_fused"):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in, self.ch_out, self.kernel_size, self.stride, groups=self.g, padding=self.padding, bias=True
            )
        kernel, bias = fuse_conv_bn(self.conv, self.norm)
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__("conv")
        self.__delattr__("norm")


class DepthwiseSeparableConv(nn.Module):
    """A depthwise 3x3 conv (with ``dilation``), a pointwise 1x1 conv, and a ReLU."""

    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=dilation, dilation=dilation, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.pointwise(self.depthwise(x)))


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation: rescale every channel by a sigmoid gate predicted from the channels' global averages."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        return x * self.fc(self.gap(x).view(b, c)).view(b, c, 1, 1)


class SCDown(nn.Module):
    """Spatial-channel decoupled downsampling: a 1x1 pointwise conv then a strided depthwise conv."""

    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = ConvNormLayerFuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayerFuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    """RepVGG block: parallel 3x3 and 1x1 conv-BN branches, foldable into one 3x3 conv for deployment."""

    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else act

    def forward(self, x):
        if hasattr(self, "conv"):
            return self.act(self.conv(x))
        return self.act(self.conv1(x) + self.conv2(x))

    def convert_to_deploy(self):
        if not hasattr(self, "conv"):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__("conv1")
        self.__delattr__("conv2")

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = fuse_conv_bn(self.conv1.conv, self.conv1.norm)
        kernel1x1, bias1x1 = fuse_conv_bn(self.conv2.conv, self.conv2.norm)
        return kernel3x3 + F.pad(kernel1x1, [1, 1, 1, 1]), bias3x3 + bias1x1


class CSPLayer(nn.Module):
    """Cross-stage partial layer: a stack of bottlenecks on one 1x1 branch, summed with a plain 1x1 branch."""

    def __init__(
        self, in_channels, out_channels, num_blocks=3, expansion=1.0, bias=False, act="silu", bottletype=VGGBlock
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayerFuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayerFuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(
            *[bottletype(hidden_channels, hidden_channels, act=get_activation(act)) for _ in range(num_blocks)]
        )
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayerFuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


class RepNCSPELAN4(nn.Module):
    """The GELAN fusion block (from YOLOv9): split, two chained CSP stages, concat everything, 1x1 out."""

    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayerFuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayerFuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayerFuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayerFuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))
