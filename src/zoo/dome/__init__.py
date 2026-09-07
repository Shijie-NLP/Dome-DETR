"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

The Dome-DETR model: ``DOME`` wires a backbone, an encoder (``HybridEncoder``, the D-FINE
baseline, or ``DomeHybridEncoder`` with DeFE and MWAS) and a decoder (``DFINETransformer``, fixed top-k
queries, ``DomeTransformer`` with PAQI, or ``MaxIoUTransformer`` with ground-truth-claimed
queries); ``DomeCriterion`` and ``HungarianMatcher`` train it
and ``DomePostProcessor`` turns its outputs into detections. Importing the package registers
all of them for the configs.
"""

from .dfine_decoder import DFINETransformer
from .dome import DOME
from .dome_criterion import DomeCriterion
from .dome_decoder import DomeTransformer
from .dome_encoder import DomeHybridEncoder
from .hybrid_encoder import HybridEncoder
from .matcher import HungarianMatcher
from .maxiou_decoder import MaxIoUTransformer
from .postprocessor import DomePostProcessor

__all__ = [
    "DFINETransformer",
    "DOME",
    "DomeCriterion",
    "DomeHybridEncoder",
    "DomePostProcessor",
    "DomeTransformer",
    "HungarianMatcher",
    "HybridEncoder",
    "MaxIoUTransformer",
]
