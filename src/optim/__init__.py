"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.

Everything the ``optimizer`` / ``lr_scheduler`` / ``lr_warmup_scheduler`` / ``ema`` / ``scaler``
entries of a config can name: the torch optimizers, schedulers and GradScaler registered under
their own names in ``optim.py``, plus the warmup schedulers and the model EMA of this package.
"""

from .ema import ModelEMA
from .optim import SGD, Adam, AdamW, CosineAnnealingLR, GradScaler, LambdaLR, MultiStepLR, OneCycleLR
from .warmup import LinearWarmup, Warmup

__all__ = [
    "SGD",
    "Adam",
    "AdamW",
    "CosineAnnealingLR",
    "GradScaler",
    "LambdaLR",
    "LinearWarmup",
    "ModelEMA",
    "MultiStepLR",
    "OneCycleLR",
    "Warmup",
]
