"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

VisDrone2019-DET, read from the Hub (https://huggingface.co/datasets/shijli/visdrone2019-det).
"""

from collections.abc import Callable

import torch

from ...core import register
from .hf_detection import HFDetection

__all__ = ["VisDroneDetection"]


# the official category ids, 0..11, by name
VISDRONE_CLASSES = [
    "ignored regions",
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
    "others",
]
IGNORED_REGIONS = 0
OTHERS = 11


@register()
class VisDroneDetection(HFDetection):
    """
    The ``detection`` config of the VisDrone2019-DET repo: 6471 train / 548 validation / 1610
    test (test-dev) images, with the official annotations whole. Labels are the official
    category ids, so the 10 scored classes are 1..10.

    The two pseudo-classes get the treatment the VisDrone toolkit gives them: ``others`` (11) is
    dropped, and ``ignored regions`` (0) become crowd boxes -- absent from the training target,
    present in the ground truth so the evaluator can discard detections that fall inside them.
    """

    REPO = "shijli/visdrone2019-det"
    CONFIG = "detection"
    CATEGORIES = [(i, name) for i, name in enumerate(VISDRONE_CLASSES) if i not in (IGNORED_REGIONS, OTHERS)]

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
        category = torch.tensor(objects["category"], dtype=torch.int64)
        boxes = torch.tensor(objects["bbox"], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # official xywh -> xyxy
        keep = category != OTHERS
        boxes, category = boxes[keep], category[keep]
        iscrowd = (category == IGNORED_REGIONS).to(torch.int64)
        return boxes, category, iscrowd
