"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

from torch.optim.lr_scheduler import LRScheduler

from ..core import register

__all__ = ["LinearWarmup", "Warmup"]


class Warmup:
    """
    Scales every parameter group's lr by ``get_warmup_factor(step)`` for the first
    ``warmup_duration`` steps, starting from the lrs the scheduler's optimizer holds at
    construction. ``step`` is called once per iteration; once ``finished`` the epoch scheduler
    takes over and this does nothing. The wrapped scheduler is not part of the state dict.
    """

    def __init__(self, lr_scheduler: LRScheduler, warmup_duration: int, last_step: int = -1) -> None:
        self.lr_scheduler = lr_scheduler
        self.warmup_end_values = [pg["lr"] for pg in lr_scheduler.optimizer.param_groups]
        self.last_step = last_step
        self.warmup_duration = warmup_duration
        self.step()

    def state_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "lr_scheduler"}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)

    def get_warmup_factor(self, step: int) -> float:
        raise NotImplementedError

    def step(self) -> None:
        self.last_step += 1
        if self.finished():
            return
        factor = self.get_warmup_factor(self.last_step)
        for pg, end_value in zip(self.lr_scheduler.optimizer.param_groups, self.warmup_end_values):
            pg["lr"] = factor * end_value

    def finished(self) -> bool:
        return self.last_step >= self.warmup_duration


@register()
class LinearWarmup(Warmup):
    """lr ramps linearly from ``1 / warmup_duration`` of its value to the full value."""

    def get_warmup_factor(self, step: int) -> float:
        return min(1.0, (step + 1) / self.warmup_duration)
