"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

AI-TOD-v2, read from the Hub (https://huggingface.co/datasets/shijli/aitod-v2).
"""

import torch

from ...core import register
from .hf_detection import HFDetection

__all__ = ["AITODDetection"]


# the official AI-TOD-v2 category ids, 0..7, by name
AITOD_CLASSES = [
    "airplane",
    "bridge",
    "storage-tank",
    "ship",
    "swimming-pool",
    "vehicle",
    "person",
    "wind-mill",
]


@register()
class AITODDetection(HFDetection):
    """
    The ``detection`` config of the AI-TOD-v2 repo: 11214 train / 2804 validation / 14018 test
    images of 800 x 800 pixels, with the official v2 annotations whole. Labels are the official
    category ids 0..7, all of them scored.

    The AI-TOD protocol trains on train + validation and reports on test; pass
    ``split: train+validation`` for that. There are no ignore regions: every box is an object,
    so the crowd flag is 0 everywhere. The 51 zero-width or zero-height boxes of the release are
    dropped from the training target by ``HFDetection`` and kept in the ground truth, where
    COCOeval scores them like any other box.
    """

    REPO = "shijli/aitod-v2"
    CONFIG = "detection"
    CATEGORIES = list(enumerate(AITOD_CLASSES))

    def parse_objects(self, objects):
        category = torch.tensor(objects["category"], dtype=torch.int64)
        boxes = torch.tensor(objects["bbox"], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # official xywh -> xyxy
        iscrowd = torch.zeros_like(category)
        return boxes, category, iscrowd

    def file_name(self, row_id):
        return row_id + ".png"
