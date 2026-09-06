"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.

Small tensor helpers shared by the detection heads.
"""

import math

import torch

__all__ = ["bias_init_with_prob", "inverse_sigmoid"]


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """The logit of ``x`` in [0, 1], with both ``x`` and ``1 - x`` clamped to ``eps`` to stay finite."""
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob: float = 0.01) -> float:
    """The bias that makes a sigmoid output ``prior_prob`` at initialisation (the focal-loss prior)."""
    return float(-math.log((1 - prior_prob) / prior_prob))
