"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
"""

import numpy as np

from ...core import register
from .coco_eval import CocoEvaluator

__all__ = ["VOCEvaluator"]


@register()
class VOCEvaluator(CocoEvaluator):
    """
    COCO-style scoring of VOC, with the VOC number on top: after the usual AP@[.5:.95] table it
    prints AP@0.5 per class and their mean, which is the figure VOC results are reported as.

    ``difficult`` objects reach the ground truth as crowd boxes (see ``VOCDetection``), which is
    how COCOeval expresses "ignore": a detection on one is neither a hit nor a false positive,
    and the box itself is never a miss. That is the VOC protocol.
    """

    def __init__(self, coco_gt, iou_types=("bbox",)):
        super().__init__(coco_gt, list(iou_types))

    def summarize(self):
        super().summarize()
        if "bbox" in self.coco_eval:
            self.summarize_per_class(self.coco_eval["bbox"])

    def summarize_per_class(self, coco_eval, iou_thr=0.5):
        # precision is [iou thresholds, recall steps, categories, area ranges, max dets]; the
        # all-areas range is index 0 and the largest max-dets setting is last. -1 marks a
        # (threshold, category) with no ground truth, which is left out of the mean.
        precision = coco_eval.eval["precision"]
        t = int(np.where(np.isclose(coco_eval.params.iouThrs, iou_thr))[0][0])
        names = {c["id"]: c["name"] for c in self.coco_gt.loadCats(self.coco_gt.getCatIds())}

        print(f"Per-class AP@{iou_thr:.2f}:")
        aps = []
        for k, cat_id in enumerate(coco_eval.params.catIds):
            p = precision[t, :, k, 0, -1]
            p = p[p > -1]
            ap = float(np.mean(p)) if p.size else float("nan")
            aps.append(ap)
            print(f"  {names.get(cat_id, cat_id):<16} {ap * 100:6.2f}")
        print(f"  {'mAP':<16} {np.nanmean(aps) * 100:6.2f}")
