"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
"""

import numpy as np
from faster_coco_eval import COCOeval_faster

from ...core import register
from .coco_eval import CocoEvaluator

__all__ = ["AITODEvaluator"]


class AITODCOCOeval(COCOeval_faster):
    """
    COCOeval with the AI-TOD settings (the aitodpycocotools protocol): up to 1500 detections per
    image, and the size bands very tiny (up to 8 px), tiny (8-16), small (16-32) and medium (over
    32) in place of COCO's small / medium / large. The summary is AP, AP50, AP75, the four
    per-band APs, AR at 1 / 100 / 1500 detections and the four per-band ARs.
    """

    def __init__(self, coco_gt, iou_type, print_function=print, separate_eval=True):
        super().__init__(coco_gt, iouType=iou_type, print_function=print_function, separate_eval=separate_eval)
        self.params.maxDets = [1, 100, 1500]
        self.params.areaRng = [
            [0**2, 1e5**2],
            [0**2, 8**2],
            [8**2, 16**2],
            [16**2, 32**2],
            [32**2, 1e5**2],
        ]
        self.params.areaRngLbl = ["all", "verytiny", "tiny", "small", "medium"]

    def summarize(self):
        if not self.eval:
            raise Exception("Please run accumulate() first")
        if self.params.iouType not in ("segm", "bbox", "boundary"):
            raise ValueError(f"AI-TOD evaluation is bbox only, got iouType={self.params.iouType}")

        max_det = self.params.maxDets[-1]
        stats = np.zeros((14,))
        stats[0] = self._summarize(1, maxDets=max_det)  # AP
        stats[1] = self._summarize(1, iouThr=0.5, maxDets=max_det)  # AP50
        stats[2] = self._summarize(1, iouThr=0.75, maxDets=max_det)  # AP75
        stats[3] = self._summarize(1, areaRng="verytiny", maxDets=max_det)
        stats[4] = self._summarize(1, areaRng="tiny", maxDets=max_det)
        stats[5] = self._summarize(1, areaRng="small", maxDets=max_det)
        stats[6] = self._summarize(1, areaRng="medium", maxDets=max_det)
        for i, max_det_i in enumerate(self.params.maxDets):  # AR@1, 100, 1500
            stats[7 + i] = self._summarize(0, maxDets=max_det_i)
        stats[10] = self._summarize(0, areaRng="verytiny", maxDets=max_det)
        stats[11] = self._summarize(0, areaRng="tiny", maxDets=max_det)
        stats[12] = self._summarize(0, areaRng="small", maxDets=max_det)
        stats[13] = self._summarize(0, areaRng="medium", maxDets=max_det)

        self.all_stats = stats
        self.stats = stats


@register()
class AITODEvaluator(CocoEvaluator):
    """COCO scoring with the AI-TOD limits and size bands; there are no ignore regions to filter."""

    def __init__(self, coco_gt, iou_types=("bbox",)):
        super().__init__(coco_gt, list(iou_types))

    def _build_coco_eval(self, iou_type):
        return AITODCOCOeval(self.coco_gt, iou_type=iou_type, print_function=print, separate_eval=True)
