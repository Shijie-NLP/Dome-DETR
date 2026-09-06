"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

Helpers shared by the transforms.
"""

from .._misc import convert_to_tv_tensor

__all__ = ["PER_OBJECT_KEYS", "restore_tv_tensors", "unpack_inputs"]

# target entries with one row per object, which are concatenated when samples are combined;
# everything else (image_id, idx, orig_size) is per image and taken from the first sample
PER_OBJECT_KEYS = ("boxes", "labels", "area", "iscrowd", "masks")


def unpack_inputs(inputs: tuple):
    """What a transform was called with: the tuple itself, or its only element when there is one."""
    return inputs if len(inputs) > 1 else inputs[0]


def restore_tv_tensors(target: dict, spatial_size) -> dict:
    """
    Re-wrap ``boxes`` (xyxy, in pixels of an image of ``spatial_size`` = (h, w)) and ``masks`` as
    tv_tensors, in place. ``torch.cat`` returns plain tensors, and the transforms downstream need
    the subclasses to know what they are looking at.
    """
    if "boxes" in target:
        target["boxes"] = convert_to_tv_tensor(target["boxes"], "boxes", box_format="xyxy", spatial_size=spatial_size)
    if "masks" in target:
        target["masks"] = convert_to_tv_tensor(target["masks"], "masks")
    return target
