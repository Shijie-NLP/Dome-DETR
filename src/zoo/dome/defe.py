"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

DeFE, the Density-Focal Extractor: a light head on the stride-4 features that predicts a
per-pixel object density map and a per-image count value. The encoder uses the map to pick the
windows MWAS attends to and the decoder to size its query budget; the criterion supervises it
with ``render_density_maps``, a Gaussian heatmap drawn from the ground-truth boxes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ...nn.blocks import ChannelAttention, DepthwiseSeparableConv

__all__ = ["DeFEStack", "LiteDeFE", "adaptive_defe_filter", "render_density_maps"]


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
        peak = density.max()
        density = torch.where(peak > 0, density / peak, density)  # no host sync on the peak
        reg_value = self.regression_head(x)
        return density, reg_value


def _kernel_sizes(px, sigma_ratio):
    """
    The Gaussian of a box side of ``px`` pixels (``[..]``, whole numbers): its sigma
    (``sigma_ratio`` times the side, at least 1), the kernel's half width (the kernel spans about
    6 sigma, rounded up to an odd width), and the divisor ``2 sigma^2``. Sigma and the width are
    computed in float64, as the Python arithmetic of the original per-box loop was.
    """
    sigma = (px.double() * sigma_ratio).clamp(min=1.0)
    width = (6 * sigma).trunc() + 1
    width = width + (width % 2 == 0)
    return width // 2, (2 * sigma**2).float()


def _axis_profiles(centre, radius, denom, length, sigma_ratio, chunk=1024):
    """
    Each box's Gaussian along one axis, sampled at the ``length`` pixel positions of the image:
    ``exp(-d^2 / denom)`` at distance ``d`` from ``centre`` within ``radius``, 0 beyond, as
    ``[..., length]``; and the sum of the same Gaussian over the full kernel ``[-radius, radius]``
    (the kernel is normalized before it is clipped to the image), as ``[...]``. The sum runs over
    offsets up to the largest kernel a box within the image can have (normalized sides at most 1),
    in chunks, so that no box count has to be read back from the device.
    """
    coords = torch.arange(length, device=centre.device, dtype=torch.float32)
    d = coords - centre[..., None]
    profile = torch.exp(-(d**2) / denom[..., None]) * (d.abs() <= radius[..., None])
    reach = int(3 * sigma_ratio * length) + 2  # half of the widest kernel: 6 sigma + 2, sigma <= sigma_ratio * length
    total = torch.zeros_like(denom)
    for start in range(-reach, reach + 1, chunk):
        offsets = torch.arange(start, min(start + chunk, reach + 1), device=centre.device, dtype=torch.float32)
        total += (torch.exp(-(offsets**2) / denom[..., None]) * (offsets.abs() <= radius[..., None])).sum(-1)
    return profile, total


def render_density_maps(boxes, size, sigma_ratio=1.2):
    """
    The ground-truth density maps of a batch: one ``[B, 1, h, w]`` map (``size`` = (h, w)) for
    ``boxes``, a list of ``[N_b, 4]`` normalized cxcywh boxes, on their device. Each map is the
    sum of one normalized Gaussian per box, centred on the box (its centre and side lengths
    truncated to whole pixels, sides at least 1) with sigma ``sigma_ratio`` times the side lengths
    and a kernel of about 6 sigma clipped to the image, scaled to a maximum of 1 (left at 0 when
    the image has no boxes).

    Rendered for the whole batch at once: the Gaussians are separable, so a map is a matrix
    product of the boxes' profiles along y and along x. The profiles multiply where the original
    per-box loop exponentiated the sum, so the values agree to float32 rounding, not bit for bit.
    """
    h, w = size
    b = len(boxes)
    device = boxes[0].device
    lengths = torch.tensor([len(x) for x in boxes])
    if int(lengths.max()) == 0:
        return torch.zeros((b, 1, h, w), device=device)
    padded = torch.nn.utils.rnn.pad_sequence(boxes, batch_first=True)  # [B, M, 4]
    valid = (torch.arange(padded.shape[1])[None, :] < lengths[:, None]).to(device, non_blocking=True)

    # the truncations of the original loop: int(centre * W), max(int(side * W), 1)
    cx, cy = (padded[..., 0] * w).trunc(), (padded[..., 1] * h).trunc()
    w_px, h_px = (padded[..., 2] * w).trunc().clamp(min=1), (padded[..., 3] * h).trunc().clamp(min=1)
    rx, denom_x = _kernel_sizes(w_px, sigma_ratio)
    ry, denom_y = _kernel_sizes(h_px, sigma_ratio)
    gx, sum_x = _axis_profiles(cx, rx, denom_x, w, sigma_ratio)  # [B, M, w], [B, M]
    gy, sum_y = _axis_profiles(cy, ry, denom_y, h, sigma_ratio)  # [B, M, h], [B, M]
    gx = gx * (valid / (sum_x * sum_y))[..., None]  # the kernel's normalization, padding zeroed
    heatmaps = torch.bmm(gy.transpose(1, 2), gx)  # [B, h, w]
    peak = heatmaps.amax((1, 2), keepdim=True)
    heatmaps = torch.where(peak > 0, heatmaps / peak, heatmaps)
    return heatmaps[:, None]


_FILTER_THRESHOLDS = (0.05, 0.04, 0.03, 0.02, 0.01, 0.0)


def adaptive_defe_filter(defe_feature, thresholds=_FILTER_THRESHOLDS):
    """
    Binarize a density map ``[B, 1, H, W]`` per image at the first of ``thresholds`` (descending)
    that some cell of the image exceeds. An image with no cell above 0 gets one random cell so
    that the window attention always has a window to work on. No host sync: the thresholds are
    picked on the device.
    """
    b, _, h, w = defe_feature.shape
    levels = torch.tensor(thresholds, dtype=defe_feature.dtype, device=defe_feature.device)
    peak = defe_feature.amax((1, 2, 3))  # [B]
    above = peak[:, None] > levels[None, :]  # [B, T], monotone along T
    chosen = levels[above.int().argmax(1)]  # the first threshold the peak exceeds (any, if none)
    mask = defe_feature > chosen[:, None, None, None]
    # the fallback: images whose map is all 0 get one random cell
    fallback = ~above.any(1)
    rows = torch.randint(h, (b,), device=defe_feature.device)
    cols = torch.randint(w, (b,), device=defe_feature.device)
    point = torch.zeros_like(mask)
    point[torch.arange(b, device=mask.device), 0, rows, cols] = True
    return torch.where(fallback[:, None, None, None], point, mask)
