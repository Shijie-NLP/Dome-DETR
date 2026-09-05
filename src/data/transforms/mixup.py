"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

MixUp for detection: blend the sample with a random other and keep the boxes of both.
"""

import random

import torch
import torchvision
import torchvision.transforms.v2 as T  # noqa: N812
from PIL import Image

from ...core import register
from .._misc import convert_to_tv_tensor
from .mosaic import PER_OBJECT_KEYS

torchvision.disable_beta_transforms_warning()


@register()
class MixUp(T.Transform):
    """
    With probability ``p``, blend the image with a random other sample (resized to the same
    size) using a weight drawn from Beta(alpha, alpha), and concatenate the two box sets.
    """

    def __init__(self, alpha=1.5, p=0.2) -> None:
        super().__init__()
        self.alpha = alpha
        self.p = p

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        image, target, dataset = inputs

        if random.random() > self.p:
            return image, target, dataset

        mix_image, mix_target = dataset.load_item(random.randint(0, len(dataset) - 1))

        # Image.blend(a, b, alpha) = a * (1 - alpha) + b * alpha, so the sample keeps weight lam
        lam = random.betavariate(self.alpha, self.alpha)
        w, h = image.size
        mixed_image = Image.blend(image, mix_image.resize((w, h)), 1 - lam)

        mixed_target = dict(target)
        for k in PER_OBJECT_KEYS:
            if k not in target:
                continue
            v = mix_target[k]
            if k == "boxes":
                # bring the other sample's boxes into this image's pixel grid
                w2, h2 = mix_image.size
                v = v.clone()
                v[:, 0::2] *= w / w2
                v[:, 1::2] *= h / h2
            mixed_target[k] = torch.cat([target[k], v], dim=0)

        # torch.cat drops the tv_tensor subclasses, and the transforms after this one need them
        if "boxes" in mixed_target:
            mixed_target["boxes"] = convert_to_tv_tensor(
                mixed_target["boxes"], "boxes", box_format="xyxy", spatial_size=[h, w]
            )
        if "masks" in mixed_target:
            mixed_target["masks"] = convert_to_tv_tensor(mixed_target["masks"], "masks")

        return mixed_image, mixed_target, dataset
