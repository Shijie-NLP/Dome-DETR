"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import random

import torch
import torchvision
import torchvision.transforms.v2 as T  # noqa: N812
from PIL import Image

from ...core import register
from .._misc import convert_to_tv_tensor

torchvision.disable_beta_transforms_warning()

# target entries with one row per object, which are concatenated when samples are combined;
# everything else (image_id, idx, orig_size) is per image and taken from the first sample
PER_OBJECT_KEYS = ("boxes", "labels", "area", "iscrowd", "masks")


@register()
class Mosaic(T.Transform):
    """
    Tile the sample with three random others into a 2x2 mosaic, jitter it with a random affine,
    and crop back to ``size``. Each tile is first resized to ``size``, so the mosaic canvas is
    twice that on each side.
    """

    def __init__(self, size, max_size=None, p=1.0) -> None:
        super().__init__()
        self.resize = T.Resize(size=size, max_size=max_size)
        self.crop = T.RandomCrop(size=max_size if max_size else size)
        self.p = p
        self.random_affine = T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.5, 1.5), fill=114)

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        image, target, dataset = inputs

        if random.random() > self.p:
            return image, target, dataset

        # the sample itself is the first tile; three random others fill the rest
        tiles = [self.resize(image, target)]
        for i in random.choices(range(len(dataset)), k=3):
            tiles.append(self.resize(*dataset.load_item(i)))
        images, targets = zip(*tiles)

        w, h = images[0].size
        offsets = [(0, 0), (w, 0), (0, h), (w, h)]
        image = Image.new(mode=images[0].mode, size=(w * 2, h * 2), color=0)
        for im, offset in zip(images, offsets):
            image.paste(im, offset)

        target = {}
        for k, v in targets[0].items():
            if k == "boxes":
                v = torch.cat([t[k] + torch.tensor(offset * 2) for t, offset in zip(targets, offsets)], dim=0)
            elif k in PER_OBJECT_KEYS:
                v = torch.cat([t[k] for t in targets], dim=0)
            target[k] = v

        if "boxes" in target:
            target["boxes"] = convert_to_tv_tensor(
                target["boxes"], "boxes", box_format="xyxy", spatial_size=[h * 2, w * 2]
            )
        if "masks" in target:
            target["masks"] = convert_to_tv_tensor(target["masks"], "masks")

        image, target = self.random_affine(image, target)
        image, target = self.crop(image, target)

        return image, target, dataset
