"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

Fine-grained Distribution Refinement (FDR), D-FINE's box regression: each box edge is predicted
as a distribution over ``reg_max + 1`` bins whose values are the non-uniform weighting function
W(n); the edge offset is the expectation under that distribution. This module holds the
weighting function, the two conversions between boxes and binned edge distances (used by the
decoder and the criterion), the Integral layer that takes the expectation, and the LQE head that
turns the distribution's sharpness into a quality score.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.nn.init as init

from ...misc.box_ops import box_xyxy_to_cxcywh
from ...nn.transformer import MLP

__all__ = ["LQE", "Integral", "bbox2distance", "distance2bbox", "translate_gt", "weighting_function"]


def weighting_function(reg_max, up, reg_scale, deploy=False):
    """
    The non-uniform weighting function W(n) for bounding box regression: ``reg_max + 1`` values
    from ``-2 * up * reg_scale`` to ``2 * up * reg_scale``, zero in the middle, spaced
    geometrically so that bins are fine near zero and coarse at the ends.

    Args:
        reg_max (int): number of discrete bins.
        up (Tensor): controls the upper bound of the sequence; the maximum offset is ±up * H / W.
        reg_scale (Tensor): controls the curvature; larger values give flatter weights near the
            centre and steeper ones at both ends.
        deploy (bool): return a constant tensor detached from ``up`` and ``reg_scale``.
    """
    upper_bound1 = abs(up[0]) * abs(reg_scale)
    upper_bound2 = upper_bound1 * 2
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left_values = [-(step**i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values = [step**i - 1 for i in range(1, reg_max // 2)]
    values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
    values = torch.cat(values, 0)
    return values.detach().clone() if deploy else values


def translate_gt(gt, reg_max, reg_scale, up):
    """
    Ground-truth edge distances as positions in the W(n) bins: for every value the index of the
    bin on its left, and the interpolation weights of the right and left bins. Values outside
    the range land fully on the first or last bin.

    Args:
        gt (Tensor): ground-truth distances, any shape (flattened to (N,)).

    Returns:
        indices (N,), weight_right (N,), weight_left (N,)
    """
    gt = gt.reshape(-1)
    function_values = weighting_function(reg_max, up, reg_scale)

    # the closest bin on the left of each value
    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
    indices = (torch.sum(diffs <= 0, dim=1) - 1).float()

    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)

    valid_idx_mask = (indices >= 0) & (indices < reg_max)
    valid_indices = indices[valid_idx_mask].long()
    left_values = function_values[valid_indices]
    right_values = function_values[valid_indices + 1]
    left_diffs = torch.abs(gt[valid_idx_mask] - left_values)
    right_diffs = torch.abs(right_values - gt[valid_idx_mask])
    weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
    weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]

    below = indices < 0
    weight_right[below] = 0.0
    weight_left[below] = 1.0
    indices[below] = 0.0

    above = indices >= reg_max
    weight_right[above] = 1.0
    weight_left[above] = 0.0
    indices[above] = reg_max - 0.1

    return indices, weight_right, weight_left


def distance2bbox(points, distance, reg_scale):
    """
    Boxes from reference boxes and predicted edge distances.

    Args:
        points (Tensor): ``(..., 4)`` reference boxes as [cx, cy, w, h].
        distance (Tensor): ``(..., 4)`` distances to the left, top, right and bottom edges, in
            units of the reference size / reg_scale, measured from half a reg_scale out.
        reg_scale: curvature of W(n).

    Returns:
        Tensor: ``(..., 4)`` boxes as [cx, cy, w, h].
    """
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)
    return box_xyxy_to_cxcywh(torch.stack([x1, y1, x2, y2], -1))


def bbox2distance(points, bbox, reg_max, reg_scale, up, eps=0.1):
    """
    The inverse of ``distance2bbox`` for training targets: ground-truth boxes as binned edge
    distances (see ``translate_gt``), clamped to ``[0, reg_max - eps]``.

    Args:
        points (Tensor): ``(n, 4)`` reference boxes as [cx, cy, w, h].
        bbox (Tensor): ``(n, 4)`` ground-truth boxes as [x1, y1, x2, y2].

    Returns:
        distances (4n,), weight_right (4n,), weight_left (4n,), all detached.
    """
    reg_scale = abs(reg_scale)
    left = (points[:, 0] - bbox[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    top = (points[:, 1] - bbox[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    right = (bbox[:, 2] - points[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    bottom = (bbox[:, 3] - points[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    four_lens = torch.stack([left, top, right, bottom], -1)
    four_lens, weight_right, weight_left = translate_gt(four_lens, reg_max, reg_scale, up)
    if reg_max is not None:
        four_lens = four_lens.clamp(min=0, max=reg_max - eps)
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()


class Integral(nn.Module):
    """
    The expectation of a binned edge distribution: ``sum_n softmax(x)[n] * W(n)`` for each of
    the four edges, where ``project`` holds W(n) (see ``weighting_function``).
    """

    def __init__(self, reg_max=32):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    """
    Location quality estimator: adds to the class scores a term predicted from the top-``k``
    probabilities (and their mean) of each edge distribution, so sharper distributions score
    higher. Initialised to zero, so it starts as a no-op.
    """

    def __init__(self, k, hidden_dim, num_layers, reg_max):
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        b, n, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(b, n, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        return scores + self.reg_conf(stat.reshape(b, n, -1))
