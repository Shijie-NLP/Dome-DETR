"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

PASCAL VOC 2012 detection, read from the Hub (https://huggingface.co/datasets/shijli/voc2012).
"""

from collections.abc import Callable

import torch

from ...core import register
from .hf_detection import HFDetection

__all__ = ["VOCDetection"]


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


@register()
class VOCDetection(HFDetection):
    """
    The ``detection`` config of the VOC 2012 repo: 5717 train / 5823 validation images, with
    the boxes exactly as the VOC XML gives them. Labels are the 20 VOC classes as 0..19.

    ``difficult`` objects are the VOC protocol's ignore set: they are left out of the training
    target and become crowd boxes in the ground truth, so a detection on one is not penalised.
    """

    REPO = "shijli/voc2012"
    CONFIG = "detection"
    CATEGORIES = list(enumerate(VOC_CLASSES))

    def __init__(
        self,
        split: str,
        repo: str = REPO,
        transforms: Callable | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
    ):
        super().__init__(
            split, repo=repo, config=self.CONFIG, transforms=transforms, revision=revision, cache_dir=cache_dir
        )

    def parse_objects(self, objects):
        boxes = torch.tensor(objects["bbox"], dtype=torch.float32).reshape(-1, 4)
        # VOC boxes are 1-indexed and inclusive on both corners; shifting the top-left corner
        # by one makes them the 0-indexed, half-open [x1, x2) boxes everything downstream assumes
        boxes[:, :2] -= 1
        labels = torch.tensor(objects["label"], dtype=torch.int64)
        iscrowd = torch.tensor(objects["difficult"], dtype=torch.int64)
        return boxes, labels, iscrowd
