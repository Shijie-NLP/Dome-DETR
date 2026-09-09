"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
"""

import numpy as np
import torch

__all__ = ["dynamic_nms", "dynamic_nms_batch"]


def _pairwise_iou(boxes):
    """``[B, N, N]`` IoU of ``[B, N, 4]`` xyxy boxes within each image (``box_ops.box_iou`` batched)."""
    area = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    lt = torch.max(boxes[:, :, None, :2], boxes[:, None, :, :2])
    rb = torch.min(boxes[:, :, None, 2:], boxes[:, None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area[:, :, None] + area[:, None, :] - inter)


def dynamic_nms_batch(boxes, scores, classes, iou_thresholds, valid=None):
    """
    Class-wise NMS with a per-box IoU threshold over a padded batch: box ``i`` suppresses a
    lower-scoring box of the same class when their IoU is at least ``iou_thresholds[i]``. Boxes
    are visited in descending score, ties by index. Returns the ``[B, N]`` mask of the kept boxes.

    Args:
        boxes: ``[B, N, 4]`` xyxy.
        scores, classes, iou_thresholds: ``[B, N]``.
        valid: ``[B, N]``, the boxes that take part (padding neither suppresses nor is kept).

    The IoU matrix and the suppression candidates (same class, IoU at least the threshold) are
    computed on the device for the whole batch; the sequential pass is a host loop over the sorted
    positions on a packed copy of the candidates, so the whole thing costs one device-to-host
    copy and one host-to-device copy however many images and classes there are.
    """
    b, n = scores.shape
    if valid is None:
        valid = torch.ones_like(scores, dtype=torch.bool)
    order = torch.sort(scores, dim=1, descending=True, stable=True).indices  # [B, N]
    boxes = boxes.gather(1, order[..., None].expand(-1, -1, 4))
    classes = classes.gather(1, order)
    thresholds = iou_thresholds.gather(1, order)
    valid = valid.gather(1, order)
    iou = _pairwise_iou(boxes)
    candidates = (classes[:, :, None] == classes[:, None, :]) & (iou >= thresholds[:, :, None]) & valid[:, None, :]
    packed = torch.cat([candidates.flatten(1), valid], dim=1).cpu().numpy()  # one copy
    candidates, valid = packed[:, : n * n].reshape(b, n, n), packed[:, n * n :]

    suppressed = np.zeros((b, n), dtype=bool)
    keep = np.zeros((b, n), dtype=bool)
    for i in range(n):
        keep[:, i] = valid[:, i] & ~suppressed[:, i]
        suppressed[:, i + 1 :] |= keep[:, i : i + 1] & candidates[:, i, i + 1 :]
    keep = torch.from_numpy(keep).to(scores.device, non_blocking=True)
    return torch.zeros_like(keep).scatter_(1, order, keep)


def dynamic_nms(boxes, scores, classes, iou_thresholds):
    """
    One image: ``dynamic_nms_batch`` on ``[N, 4]`` boxes and ``[N]`` scores, classes and
    thresholds. Returns the indices of the kept boxes, in ascending order.
    """
    keep = dynamic_nms_batch(boxes[None], scores[None], classes[None], iou_thresholds[None])[0]
    return torch.nonzero(keep, as_tuple=True)[0]
