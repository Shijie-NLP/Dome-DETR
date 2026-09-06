"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import Any

import PIL
import PIL.Image
import torch
import torchvision
import torchvision.transforms.v2 as T  # noqa: N812
import torchvision.transforms.v2.functional as F  # noqa: N812

from ...core import register
from .._misc import BoundingBoxes, Image, Mask, Video, convert_to_tv_tensor
from ._utils import unpack_inputs

# torchvision transforms exposed to the yaml configs under their own names
RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
SanitizeBoundingBoxes = register()(T.SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def forward(self, *inputs):
        return unpack_inputs(inputs)


@register()
class PadToSize(T.Pad):
    """Pad the bottom and right edges up to ``size`` (w, h), and record the padding in the target."""

    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def __init__(self, size, fill=0, padding_mode="constant") -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _get_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        h, w = F.get_size(flat_inputs[0])
        self.padding = [0, 0, self.size[0] - w, self.size[1] - h]
        return dict(padding=self.padding)

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        return F.pad(inpt, padding=params["padding"], fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]["padding"] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    """torchvision's RandomIoUCrop, applied with probability ``p``."""

    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2,
        sampler_options: list[float] | None = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return unpack_inputs(inputs)
        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    """Change the box format (e.g. xyxy -> cxcywh) and optionally normalize by the image size."""

    _transformed_types = (BoundingBoxes,)

    def __init__(self, fmt="", normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        spatial_size = inpt.canvas_size
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key="boxes", box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            # divides by [w, h, w, h]; the result is a plain tensor, which is what the model consumes
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    """PIL image -> tv_tensors.Image, as float in [0, 1] by default."""

    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == "float32":
            inpt = inpt.float()
        if self.scale:
            inpt = inpt / 255.0
        return Image(inpt)
