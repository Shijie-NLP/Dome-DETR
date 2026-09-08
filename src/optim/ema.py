"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn

from ..core import register
from ..misc import dist_utils

__all__ = ["ModelEMA"]


@register()
class ModelEMA:
    """
    An exponential moving average of a model's state dict (parameters and buffers), after
    https://github.com/rwightman/pytorch-image-models.

    ``module`` is a frozen deep copy of the model in eval mode; ``update`` moves every floating
    point entry towards the live model's by ``1 - decay``. For the first ``warmups`` updates the
    effective decay ramps up as ``decay * (1 - exp(-updates / warmups))`` so the average can
    follow the fast early changes. ``decay`` can be changed after construction (the solver does
    so when it restarts the EMA between training stages) and takes effect on the next update.

    Create it after the model is on its device and before it is wrapped for distributed
    training, or pass the wrapped model: the copy is taken from ``de_parallel(model)`` either way.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmups: int = 1000):
        self.module = deepcopy(dist_utils.de_parallel(model)).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

        self.decay = decay
        self.warmups = warmups
        self.updates = 0  # number of EMA updates

    def effective_decay(self) -> float:
        if self.warmups == 0:
            return self.decay
        return self.decay * (1 - math.exp(-self.updates / self.warmups))

    def _pairs(self, model: nn.Module) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        The floating point entries of the average's and the model's state dicts, paired by name
        and kept: the entries are views of the parameters and buffers, which loading a state dict
        or an optimizer step update in place, so the lists stay valid until the module is moved.
        """
        model = dist_utils.de_parallel(model)
        if getattr(self, "_paired_model", None) is not model:
            msd = model.state_dict()
            names = [k for k, v in self.module.state_dict().items() if v.dtype.is_floating_point]
            ema_sd = self.module.state_dict()
            self._pairs_cache = ([ema_sd[k] for k in names], [msd[k] for k in names])
            self._paired_model = model
        return self._pairs_cache

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.effective_decay()
        ema, live = self._pairs(model)
        torch._foreach_lerp_(ema, live, 1 - d)  # ema = ema + (1 - d) * (live - ema), one launch per dtype

    def to(self, *args, **kwargs):
        self.module = self.module.to(*args, **kwargs)
        self._paired_model = None  # moving replaces the tensors
        return self

    def state_dict(self) -> dict:
        return dict(module=self.module.state_dict(), updates=self.updates)

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        self.module.load_state_dict(state["module"], strict=strict)
        if "updates" in state:
            self.updates = state["updates"]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(decay={self.decay}, warmups={self.warmups}, updates={self.updates})"
