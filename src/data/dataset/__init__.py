"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

Datasets are read straight from the Hugging Face Hub: one ``HFDetection`` subclass per repo,
and one evaluator per dataset protocol.
"""

__all__ = [
    "HFDetection",
    "get_coco_api_from_dataset",
    "AITOD_CLASSES",
    "AITODDetection",
    "AITODEvaluator",
    "VOC_CLASSES",
    "VOCDetection",
    "VOCEvaluator",
    "VISDRONE_CLASSES",
    "VisDroneDetection",
    "VisDroneEvaluator",
]

from .aitod_detection import AITOD_CLASSES, AITODDetection
from .aitod_eval import AITODEvaluator
from .hf_detection import HFDetection, get_coco_api_from_dataset
from .visdrone_detection import VISDRONE_CLASSES, VisDroneDetection
from .visdrone_eval import VisDroneEvaluator
from .voc_detection import VOC_CLASSES, VOCDetection
from .voc_eval import VOCEvaluator
