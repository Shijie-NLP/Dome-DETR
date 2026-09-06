"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.

The torchvision tv_tensor types the data pipeline works with, and the one helper that wraps a
plain tensor into one. torchvision >= 0.17 (the stable transforms.v2 API) is required.
"""

from torch import Tensor
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image, Mask, Video

__all__ = ["BoundingBoxFormat", "BoundingBoxes", "Image", "Mask", "Video", "convert_to_tv_tensor"]


def convert_to_tv_tensor(tensor: Tensor, key: str, box_format: str = "xyxy", spatial_size=None) -> Tensor:
    """
    ``tensor`` as the tv_tensor a target entry ``key`` holds: ``boxes`` become ``BoundingBoxes``
    in ``box_format`` on a canvas of ``spatial_size`` = (h, w); ``masks`` become ``Mask``.
    torchvision's transforms only move what they can recognise, so anything that rebuilds these
    entries from plain tensors (``torch.cat`` after mixing samples, say) has to re-wrap them.
    """
    if key == "boxes":
        return BoundingBoxes(tensor, format=BoundingBoxFormat[box_format.upper()], canvas_size=spatial_size)
    if key == "masks":
        return Mask(tensor)
    raise ValueError(f"only 'boxes' and 'masks' have a tv_tensor type, got {key!r}")
