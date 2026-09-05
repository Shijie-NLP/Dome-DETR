"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

from ._solver import BaseSolver
from .det_solver import DetSolver

# solver per `task` key of the config
TASKS: dict[str, type[BaseSolver]] = {
    "detection": DetSolver,
}

__all__ = ["TASKS", "BaseSolver", "DetSolver"]
