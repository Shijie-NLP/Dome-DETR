"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

The Dome-DETR model: ``DOME`` wires a backbone, the ``HybridEncoder`` (with DeFE and MWAS) and
the ``DomeTransformer`` decoder (with PAQI); ``DomeCriterion`` and ``HungarianMatcher`` train it
and ``DomePostProcessor`` turns its outputs into detections. Importing the package registers
all of them for the configs.
"""

from .dome import DOME
from .dome_criterion import DomeCriterion
from .dome_decoder import DomeTransformer
from .hybrid_encoder import HybridEncoder
from .matcher import HungarianMatcher
from .postprocessor import DomePostProcessor

__all__ = ["DOME", "DomeCriterion", "DomePostProcessor", "DomeTransformer", "HungarianMatcher", "HybridEncoder"]
