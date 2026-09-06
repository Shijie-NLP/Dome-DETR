"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

from typing import Any

import torch.nn as nn
import torchvision.transforms.v2 as T  # noqa: N812

from ...core import GLOBAL_CONFIG, create, register
from ._transforms import EmptyTransform
from ._utils import unpack_inputs

__all__ = ["Compose"]


@register()
class Compose(T.Compose):
    """
    The transform pipeline of a dataset, built from the yaml's ``ops`` list: each entry is either
    an already built transform or a ``{type: <registered name>, ...}`` dict. No ``ops`` gives a
    pipeline that returns its input.

    ``policy`` switches a subset of the transforms off partway through training, by class name:

    - ``{name: default}``: run everything, always;
    - ``{name: stop_epoch, epoch: E, ops: [...]}``: skip ``ops`` once the dataset's epoch is ``E``
      or later (the dataset arrives as the last element of the sample);
    - ``{name: stop_sample, sample: N, ops: [...]}``: skip ``ops`` after this pipeline has
      processed ``N`` samples (counted per process, so per dataloader worker).
    """

    POLICIES = ("default", "stop_epoch", "stop_sample")

    def __init__(self, ops: list[dict | nn.Module] | None, policy: dict | None = None) -> None:
        transforms = [self._build(op) for op in ops] if ops else [EmptyTransform()]
        super().__init__(transforms=transforms)

        self.policy = {"name": "default"} if policy is None else policy
        if self.policy["name"] not in self.POLICIES:
            raise ValueError(f"unknown transform policy {self.policy['name']!r}; known: {self.POLICIES}")
        self.global_samples = 0

    @staticmethod
    def _build(op: dict | nn.Module) -> nn.Module:
        if isinstance(op, nn.Module):
            return op
        if isinstance(op, dict):
            args = {k: v for k, v in op.items() if k != "type"}
            return create(op["type"], GLOBAL_CONFIG, **args)
        raise ValueError(f"a transform is a module or a {{type: ...}} dict, got {type(op)}")

    def _skipped(self, sample) -> set[str]:
        """The class names of the transforms the policy switches off for this sample."""
        name = self.policy["name"]
        if name == "stop_epoch" and sample[-1].epoch >= self.policy["epoch"]:
            return set(self.policy["ops"])
        if name == "stop_sample" and self.global_samples >= self.policy["sample"]:
            return set(self.policy["ops"])
        return set()

    def forward(self, *inputs: Any) -> Any:
        sample = unpack_inputs(inputs)
        skipped = self._skipped(sample)
        for transform in self.transforms:
            if type(transform).__name__ not in skipped:
                sample = transform(sample)
        self.global_samples += 1
        return sample
