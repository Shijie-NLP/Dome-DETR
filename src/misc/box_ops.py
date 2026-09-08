"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.

Box utilities. Boxes are ``[..., 4]`` tensors, ``cxcywh`` (centre and size) or ``xyxy``
(corners); the pairwise functions take ``[N, 4]`` and ``[M, 4]`` and return ``[N, M]``, the
elementwise ones take two broadcastable ``[..., 4]`` and return ``[...]`` (so ``[N, 1, 4]`` against
``[M, 4]`` is a pairwise ``[N, M]`` too).
"""

import torch
from torch import Tensor
from torchvision.ops.boxes import box_area

__all__ = [
    "box_cxcywh_to_xyxy",
    "box_iou",
    "box_xyxy_to_cxcywh",
    "elementwise_box_iou",
    "elementwise_generalized_box_iou",
    "gaussian_box_similarity",
    "generalized_box_iou",
]


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    w, h = w.clamp(min=0.0), h.clamp(min=0.0)
    return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    """Pairwise IoU and union area of xyxy boxes, ``[N, M]`` each."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Pairwise generalized IoU (https://giou.stanford.edu/) of xyxy boxes, ``[N, M]``. The boxes
    must be well formed (``x2 >= x1``, ``y2 >= y1``; ``box_cxcywh_to_xyxy`` guarantees it), or
    the result is inf / nan; there is no check, which would be a host sync per call.
    """
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area


def _area(boxes: Tensor) -> Tensor:
    """Area of xyxy boxes ``[..., 4]``, as torchvision's ``box_area`` but over any leading dims."""
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def elementwise_box_iou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    """IoU and union area of the i-th box of each set, ``[...]`` each (the diagonal of ``box_iou``)."""
    area1 = _area(boxes1)
    area2 = _area(boxes2)
    lt = torch.max(boxes1[..., :2], boxes2[..., :2])
    rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1 + area2 - inter
    return inter / union, union


def gaussian_box_similarity(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-6) -> Tensor:
    """
    One minus the Hellinger distance between cxcywh boxes taken as the Gaussians
    ``N((cx, cy), diag((w/2)^2, (h/2)^2))``, ``[...]`` in [0, 1] over broadcastable ``[..., 4]``
    inputs: 1 for identical boxes, parameter-free, scale-invariant, smooth in the offset and
    defined for boxes that do not overlap; about three times less sensitive to a small offset
    than IoU. Per axis the Bhattacharyya distance is ``dc^2 / (w1^2 + w2^2) + ln((w1^2 + w2^2) /
    (2 w1 w2)) / 2``, the coefficient ``exp(-distance)`` and the Hellinger distance
    ``sqrt(1 - coefficient)``.
    """
    size1, size2 = boxes1[..., 2:].clamp(min=eps), boxes2[..., 2:].clamp(min=eps)
    size_sq = size1**2 + size2**2
    centre = (boxes1[..., :2] - boxes2[..., :2]) ** 2 / size_sq
    size = 0.5 * torch.log(size_sq / (2 * size1 * size2))
    coefficient = torch.exp(-(centre + size).sum(-1))
    # the square root's gradient is infinite at 0 (identical boxes): floored at eps^2, so the
    # similarity of identical boxes is 1 - eps and its gradient there 0
    return 1 - (1 - coefficient).clamp(min=eps**2).sqrt()


def elementwise_generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Generalized IoU of the i-th box of each set, ``[...]`` (the diagonal of ``generalized_box_iou``;
    well-formed boxes, as there).
    """
    iou, union = elementwise_box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[..., :2], boxes2[..., :2])
    rb = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]
    return iou - (area - union) / area
