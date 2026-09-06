"""
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

Importing ``src`` imports every package that registers classes for the yaml configs (datasets,
transforms, backbones, optimizers, the Dome model zoo), so ``core.create`` can find them.
"""

from . import core, data, misc, nn, optim, solver, zoo

__all__ = ["core", "data", "misc", "nn", "optim", "solver", "zoo"]
