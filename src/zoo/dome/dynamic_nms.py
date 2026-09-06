"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
"""

import torch

from .box_ops import box_iou

__all__ = ["dynamic_nms"]


def dynamic_nms(boxes, scores, classes, iou_thresholds):
    """
    Class-wise NMS with a per-box IoU threshold: box ``i`` suppresses a lower-scoring box of
    the same class when their IoU is at least ``iou_thresholds[i]``. Returns the indices of the
    kept boxes, in ascending order.

    Args:
        boxes: ``[N, 4]`` xyxy.
        scores, classes, iou_thresholds: ``[N]``.
    """
    keep_mask = torch.zeros_like(classes, dtype=torch.bool)
    for cls in classes.unique():
        cls_indices = torch.nonzero(classes == cls, as_tuple=True)[0]
        keep_cls = _single_class_dynamic_nms(boxes[cls_indices], scores[cls_indices], iou_thresholds[cls_indices])
        keep_mask[cls_indices[keep_cls]] = True
    return torch.nonzero(keep_mask, as_tuple=True)[0]


def _single_class_dynamic_nms(boxes, scores, iou_thresholds):
    """One pass over the boxes in score order, with the full IoU matrix computed up front."""
    order = scores.argsort(descending=True)
    boxes = boxes[order]
    thresholds = iou_thresholds[order]
    iou_matrix, _ = box_iou(boxes, boxes)

    num = boxes.shape[0]
    keep_flags = torch.ones(num, dtype=torch.bool, device=boxes.device)
    keep = []
    for i in range(num):
        if not keep_flags[i]:
            continue
        keep.append(i)
        if i < num - 1:
            # every later box that overlaps box i by at least its threshold goes
            keep_flags[i + 1 :] &= ~(iou_matrix[i, i + 1 :] >= thresholds[i])
    return order[torch.tensor(keep, device=boxes.device)]
