"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import random
from itertools import product

import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils.data as data

from ..core import register

__all__ = [
    "BaseCollateFunction",
    "BatchImageCollateFunction",
    "DataLoader",
    "generate_scales",
]


class EpochAware:
    """Something the solver tells the current epoch to; ``epoch`` is -1 until it does."""

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        return getattr(self, "_epoch", -1)


@register()
class DataLoader(data.DataLoader, EpochAware):
    """
    torch's DataLoader with the two things the training loop needs on top: ``set_epoch``, which
    forwards the epoch to the dataset and the collate function (both may change behaviour with
    it), and a ``shuffle`` flag that survives construction so the distributed sampler can be
    built with the same setting.
    """

    __inject__ = ["dataset", "collate_fn"]

    def __repr__(self) -> str:
        fields = ["dataset", "batch_size", "num_workers", "drop_last", "collate_fn"]
        body = "".join(f"\n    {n}: {getattr(self, n)}" for n in fields)
        return f"{self.__class__.__name__}({body}\n)"

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle: bool) -> None:
        if not isinstance(shuffle, bool):
            raise TypeError(f"shuffle must be a bool, got {type(shuffle)}")
        self._shuffle = shuffle


class BaseCollateFunction(EpochAware):
    def __call__(self, items):
        raise NotImplementedError


def _axis_scales(size: int, repeat: int, step: int) -> list[int]:
    """
    The multi-scale sizes for one image side: ``size`` rounded down to a multiple of ``step``,
    then every multiple of ``step`` from 0.75x to 1.25x of it, with the base size itself listed
    ``repeat`` times so that it is drawn more often. Sizes run upwards to the base size and then
    downwards from the top, which is the order the released models were trained with.
    """
    size = (size // step) * step or step
    low = list(range(-(-int(size * 0.75) // step) * step, size + 1, step))  # ceil to a multiple of step
    top = (int(size * 1.25) // step) * step
    high = list(range(top, size - 1, -step)) if top >= size + step else []
    return low + [size] * repeat + high


def generate_scales(base_size, base_size_repeat: int, window_size: int) -> list[tuple[int, int]]:
    """
    The ``(h, w)`` sizes a multi-scale batch is resized to. Every size is a multiple of
    ``4 * window_size`` so that the stride-4 feature map divides into whole MWAS windows. A scalar
    ``base_size`` gives square sizes; an ``(h, w)`` pair gives the cartesian product of the two
    sides' scales.
    """
    step = 4 * window_size
    if isinstance(base_size, (list, tuple)):
        h, w = base_size
        return list(product(_axis_scales(h, base_size_repeat, step), _axis_scales(w, base_size_repeat, step)))
    return [(s, s) for s in _axis_scales(base_size, base_size_repeat, step)]


@register()
class BatchImageCollateFunction(BaseCollateFunction):
    """
    Stack the images of a batch and, while ``epoch < stop_epoch`` and ``base_size_repeat`` is
    given, resize the whole batch to a size drawn from ``generate_scales``. The solver also reads
    ``stop_epoch`` as the boundary between its two training stages.

    ``mwas_window_size`` should match the encoder's, so that every drawn size divides into whole
    windows; the default is a multiple of the configured windows, which is also enough.
    """

    def __init__(self, stop_epoch=None, base_size=(640, 640), base_size_repeat=None, mwas_window_size=20) -> None:
        super().__init__()
        self.base_size = base_size
        self.window_size = mwas_window_size
        self.scales = (
            generate_scales(base_size, base_size_repeat, mwas_window_size) if base_size_repeat is not None else None
        )
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        if self.scales is not None and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
            if "masks" in targets[0]:
                for tg in targets:
                    tg["masks"] = F.interpolate(tg["masks"], size=sz, mode="nearest")

        return images, targets
