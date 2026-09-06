"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""

import contextlib
import copy
import os

import faster_coco_eval.core.mask as mask_util
import numpy as np
import torch
from faster_coco_eval import COCO, COCOeval_faster

from ...core import register
from ...misc import dist_utils

__all__ = ["CocoEvaluator"]


@register()
class CocoEvaluator:
    """
    Accumulates detections over a validation pass and scores them with COCOeval.

    ``update`` is called once per batch with ``{image_id: {"boxes", "scores", "labels", ...}}``
    (boxes xyxy in original-image pixels); ``synchronize_between_processes`` gathers the
    per-image results across ranks; ``accumulate`` and ``summarize`` produce the usual table.
    Dataset-specific evaluators subclass this and override ``_build_coco_eval`` for other
    COCOeval parameters, or ``filter_prediction`` to drop detections before scoring.
    """

    def __init__(self, coco_gt: COCO, iou_types):
        assert isinstance(iou_types, (list, tuple))
        self.coco_gt = copy.deepcopy(coco_gt)
        self.iou_types = list(iou_types)
        self.cleanup()

    def _build_coco_eval(self, iou_type):
        """The COCOeval for one iou type. Subclasses swap in their own parameters here."""
        return COCOeval_faster(self.coco_gt, iouType=iou_type, print_function=print, separate_eval=True)

    def cleanup(self):
        """Forget every detection seen so far, so the same evaluator can score the next pass."""
        self.coco_eval = {iou_type: self._build_coco_eval(iou_type) for iou_type in self.iou_types}
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]

            # suppress pycocotools prints
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                coco_eval.cocoDt = self.coco_gt.loadRes(results) if results else COCO()
                coco_eval.params.imgIds = list(img_ids)
                coco_eval.evaluate()

            self.eval_imgs[iou_type].append(
                np.array(coco_eval._evalImgs_cpp).reshape(
                    len(coco_eval.params.catIds),
                    len(coco_eval.params.areaRng),
                    len(coco_eval.params.imgIds),
                )
            )

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type])

            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        prepare = {
            "bbox": self.prepare_for_coco_detection,
            "segm": self.prepare_for_coco_segmentation,
            "keypoints": self.prepare_for_coco_keypoint,
        }
        if iou_type not in prepare:
            raise ValueError(f"Unknown iou type {iou_type}")
        return prepare[iou_type](predictions)

    def filter_prediction(self, image_id, prediction: dict) -> dict:
        """One image's ``{"boxes", "scores", "labels", ...}`` with whatever should not be scored removed."""
        return prediction

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            prediction = self.filter_prediction(original_id, prediction)
            if len(prediction["boxes"]) == 0:
                continue

            boxes = convert_to_xywh(prediction["boxes"]).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                {"image_id": original_id, "category_id": labels[k], "bbox": box, "score": scores[k]}
                for k, box in enumerate(boxes)
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            masks = prediction["masks"] > 0.5
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                {"image_id": original_id, "category_id": labels[k], "segmentation": rle, "score": scores[k]}
                for k, rle in enumerate(rles)
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"].flatten(start_dim=1).tolist()

            coco_results.extend(
                {"image_id": original_id, "category_id": labels[k], "keypoints": keypoint, "score": scores[k]}
                for k, keypoint in enumerate(keypoints)
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    """
    Gather the per-image evaluation of every rank into one flat list, in the layout
    ``COCOeval.accumulate`` reads it back from (``_evalImgs_cpp`` indexed by category, area
    range, image).
    """
    merged_img_ids = [i for rank_ids in dist_utils.all_gather(img_ids) for i in rank_ids]
    merged_eval_imgs = [e for rank_evals in dist_utils.all_gather(eval_imgs) for e in rank_evals]

    merged_img_ids = np.unique(np.array(merged_img_ids))  # unique, sorted
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2).ravel()

    return merged_img_ids.tolist(), merged_eval_imgs.tolist()
