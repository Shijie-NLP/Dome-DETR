"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.

Importing this package registers every transform a yaml ``ops`` list can name.
"""

from ._transforms import (
    ConvertBoxes,
    ConvertPILImage,
    EmptyTransform,
    Normalize,
    PadToSize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomIoUCrop,
    RandomPhotometricDistort,
    RandomZoomOut,
    Resize,
    SanitizeBoundingBoxes,
)
from .container import Compose
from .mixup import MixUp
from .mosaic import Mosaic

__all__ = [
    "Compose",
    "ConvertBoxes",
    "ConvertPILImage",
    "EmptyTransform",
    "MixUp",
    "Mosaic",
    "Normalize",
    "PadToSize",
    "RandomCrop",
    "RandomHorizontalFlip",
    "RandomIoUCrop",
    "RandomPhotometricDistort",
    "RandomZoomOut",
    "Resize",
    "SanitizeBoundingBoxes",
]
