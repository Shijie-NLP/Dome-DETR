"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

DeFE, the Density-Focal Extractor: a light head on the stride-4 features that predicts a
per-pixel object density map and a per-image count value. The encoder uses the map to pick the
windows MWAS attends to and the decoder to size its query budget; the criterion supervises it
with ``render_density_map``, a Gaussian heatmap drawn from the ground-truth boxes.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ...nn.blocks import ChannelAttention, DepthwiseSeparableConv

__all__ = ["DeFEStack", "LiteDeFE", "adaptive_defe_filter", "render_density_map"]


class DeFEStack(nn.Module):
    """
    The DeFE trunk: depthwise-separable 3x3 convs with dilations ``dilations`` (multi-scale
    context at little cost), each followed by BatchNorm, with a channel attention block after
    the one at index ``attention_after``.
    """

    def __init__(self, channels=256, dilations=(1, 2, 3, 1, 1), attention_after=2):
        super().__init__()
        layers = []
        for idx, dilation in enumerate(dilations):
            layers += [DepthwiseSeparableConv(channels, channels, dilation), nn.BatchNorm2d(channels)]
            if idx == attention_after:
                layers.append(ChannelAttention(channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LiteDeFE(nn.Module):
    """
    Density and count prediction from a ``[B, C, H, W]`` feature map. The input is projected,
    pooled 2x and passed through ``DeFEStack``; the density head upsamples back to ``[B, 1, H, W]``
    and the count head pools to one sigmoid value per image.

    The density map is divided by its maximum over the whole batch (kept as trained: not per
    image), so it is 0-1 normalized with at least one cell at 1.
    """

    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1), nn.AvgPool2d(kernel_size=2))
        self.defe = DeFEStack(channels)
        self.density_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid(),
        )
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        x = self.defe(self.conv1(features))
        density = F.interpolate(self.density_head(x), scale_factor=2, mode="bilinear", align_corners=False)
        if density.max() > 0:
            density = density / density.max()
        reg_value = self.regression_head(x)
        return density, reg_value


def _gaussian_kernel(sigma_x, sigma_y):
    """A normalized 2D Gaussian on an odd grid of about 6 sigma per side."""
    sigma_x, sigma_y = max(sigma_x, 0.1), max(sigma_y, 0.1)
    kernel_w = int(6 * sigma_x) + 1
    kernel_h = int(6 * sigma_y) + 1
    kernel_w += kernel_w % 2 == 0
    kernel_h += kernel_h % 2 == 0

    x = torch.arange(kernel_w, dtype=torch.float32) - (kernel_w // 2)
    y = torch.arange(kernel_h, dtype=torch.float32) - (kernel_h // 2)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    kernel = torch.exp(-(xx**2 / (2 * sigma_x**2) + yy**2 / (2 * sigma_y**2)))
    kernel_sum = kernel.sum()
    return kernel / kernel_sum if kernel_sum > 0 else kernel


def render_density_map(boxes, size, sigma_ratio=1.2):
    """
    The ground-truth density map for ``boxes`` (normalized cxcywh) on an image of ``size`` =
    (h, w): one normalized Gaussian per box, centred on the box and as wide as
    ``sigma_ratio`` times its side lengths, summed and scaled to a maximum of 1. Returns
    ``[1, h, w]`` on the CPU.
    """
    H, W = size  # noqa: N806
    heatmap = torch.zeros((H, W), dtype=torch.float32)

    for x_center, y_center, width, height in boxes:
        cx, cy = int(x_center * W), int(y_center * H)
        w_px, h_px = max(int(width * W), 1), max(int(height * H), 1)
        kernel = _gaussian_kernel(max(w_px * sigma_ratio, 1.0), max(h_px * sigma_ratio, 1.0))
        if kernel.numel() == 0:
            continue
        k_h, k_w = kernel.shape
        radius_x, radius_y = k_w // 2, k_h // 2

        # the part of the kernel that lands inside the image
        x_start, y_start = max(cx - radius_x, 0), max(cy - radius_y, 0)
        x_end, y_end = min(cx + radius_x + 1, W), min(cy + radius_y + 1, H)
        k_start_x = max(radius_x - (cx - x_start), 0)
        k_start_y = max(radius_y - (cy - y_start), 0)
        k_end_x = k_w - max((cx + radius_x + 1) - x_end, 0)
        k_end_y = k_h - max((cy + radius_y + 1) - y_end, 0)
        patch = kernel[k_start_y:k_end_y, k_start_x:k_end_x]
        if patch.numel() == 0:
            continue
        patch = patch[: y_end - y_start, : x_end - x_start]
        heatmap[y_start:y_end, x_start:x_end] += patch

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap.unsqueeze(0)


def adaptive_defe_filter(defe_feature, init_thresh=0.05, step=0.01):
    """
    Binarize a density map [B, 1, H, W] per image, lowering the threshold from ``init_thresh``
    in steps of ``step`` until something passes. An all-zero map gets one random cell so that
    the window attention always has a window to work on.
    """
    final_mask = torch.zeros_like(defe_feature, dtype=torch.bool)
    for b in range(defe_feature.shape[0]):
        single_feat = defe_feature[b : b + 1]
        current_thresh = init_thresh
        while current_thresh >= 0:
            mask = single_feat > current_thresh
            if mask.any():
                final_mask[b : b + 1] = mask
                break
            current_thresh = round(current_thresh - step, 2)
        else:
            final_mask[
                b, :, random.randint(0, single_feat.shape[2] - 1), random.randint(0, single_feat.shape[3] - 1)
            ] = True
            print(f"Batch {b}: No valid region found, use random point enhancement")
    return final_mask
