"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
"""

import numpy as np
import torch
from faster_coco_eval import COCOeval_faster

from ...core import register
from .coco_eval import CocoEvaluator, convert_to_xywh

__all__ = ["VisDroneEvaluator"]


class VisDroneCOCOeval(COCOeval_faster):
    """COCOeval with the VisDrone settings: up to 500 detections per image, and AR at 1/10/100/500."""

    def __init__(self, coco_gt, iou_type, print_function=print, separate_eval=True):
        super().__init__(coco_gt, iouType=iou_type, print_function=print_function, separate_eval=separate_eval)
        self.params.maxDets = [1, 10, 100, 500]
        self.params.areaRng = [
            [0**2, 1e5**2],
            [0**2, 32**2],
            [32**2, 96**2],
            [96**2, 1e5**2],
        ]
        self.areaRngLbl = ["all", "small", "medium", "large"]

    def summarize(self):
        if not self.eval:
            raise Exception("Please run accumulate() first")
        if self.params.iouType not in ("segm", "bbox", "boundary"):
            raise ValueError(f"VisDrone evaluation is bbox only, got iouType={self.params.iouType}")

        max_det = self.params.maxDets[-1]
        stats = np.zeros((13,))
        stats[0] = self._summarize(1, maxDets=max_det)  # AP
        stats[1] = self._summarize(1, iouThr=0.5, maxDets=max_det)  # AP50
        stats[2] = self._summarize(1, iouThr=0.75, maxDets=max_det)  # AP75
        stats[3] = self._summarize(1, areaRng="small", maxDets=max_det)
        stats[4] = self._summarize(1, areaRng="medium", maxDets=max_det)
        stats[5] = self._summarize(1, areaRng="large", maxDets=max_det)
        for i, max_det_i in enumerate(self.params.maxDets):  # AR@1, 10, 100, 500
            stats[6 + i] = self._summarize(0, maxDets=max_det_i)
        stats[10] = self._summarize(0, areaRng="small", maxDets=max_det)
        stats[11] = self._summarize(0, areaRng="medium", maxDets=max_det)
        stats[12] = self._summarize(0, areaRng="large", maxDets=max_det)

        self.all_stats = stats
        self.stats = stats[:12]


@register()
class VisDroneEvaluator(CocoEvaluator):
    """
    The VisDrone protocol on top of COCO scoring: detections that fall inside an ignored region
    are discarded before evaluation, the way the official toolkit does it, and the evaluation
    runs with the VisDrone limits (500 detections per image).

    Ignored regions are the crowd boxes of the ground truth (see ``VisDroneDetection``). A
    detection is inside one when the region covers at least ``ignore_iof_threshold`` of the
    detection's own area.
    """

    def __init__(self, coco_gt, iou_types=("bbox",), ignore_iof_threshold=0.5):
        super().__init__(coco_gt, list(iou_types))
        self.ignore_iof_threshold = ignore_iof_threshold
        self.ignore_regions = self._collect_ignore_regions(self.coco_gt)

    def _build_coco_eval(self, iou_type):
        return VisDroneCOCOeval(self.coco_gt, iou_type=iou_type, print_function=print, separate_eval=True)

    @staticmethod
    def _collect_ignore_regions(coco_gt):
        """Per image, the ignore regions as an ``[N, 4]`` xyxy tensor; images without any are absent."""
        ignore_regions = {}
        for image_id, annotations in coco_gt.imgToAnns.items():
            boxes = []
            for ann in annotations:
                if not (ann.get("iscrowd", 0) == 1 or ann.get("ignore", 0) == 1):
                    continue
                x, y, w, h = ann["bbox"]
                if w <= 0 or h <= 0:
                    continue
                boxes.append([x, y, x + w, y + h])
            if boxes:
                ignore_regions[image_id] = torch.tensor(boxes, dtype=torch.float32)
        return ignore_regions

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            scores = prediction["scores"]
            labels = prediction["labels"]

            ignore_boxes = self.ignore_regions.get(original_id)
            if ignore_boxes is not None and len(boxes) > 0:
                keep = ~detections_in_ignore_regions(boxes, ignore_boxes.to(boxes.device), self.ignore_iof_threshold)
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            if len(boxes) == 0:
                continue

            boxes = convert_to_xywh(boxes).tolist()
            scores = scores.tolist()
            labels = labels.tolist()
            coco_results.extend(
                {"image_id": original_id, "category_id": labels[k], "bbox": box, "score": scores[k]}
                for k, box in enumerate(boxes)
            )
        return coco_results


def detections_in_ignore_regions(boxes, ignore_boxes, iof_threshold):
    """Which of ``boxes`` (xyxy) have at least ``iof_threshold`` of their area inside some ignore box."""
    if boxes.numel() == 0 or ignore_boxes.numel() == 0:
        return torch.zeros((boxes.shape[0],), dtype=torch.bool, device=boxes.device)

    top_left = torch.maximum(boxes[:, None, :2], ignore_boxes[None, :, :2])
    bottom_right = torch.minimum(boxes[:, None, 2:], ignore_boxes[None, :, 2:])
    intersection_wh = (bottom_right - top_left).clamp(min=0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    box_area = ((boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6) * (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)).unsqueeze(1)
    iof = intersection / box_area
    return iof.max(dim=1).values >= iof_threshold
