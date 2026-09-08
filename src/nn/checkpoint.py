"""
Activation checkpointing for modules with BatchNorm. ``torch.utils.checkpoint`` runs the module's
forward a second time in the backward pass to rebuild the activations it did not keep; a plain
BatchNorm would then fold the same batch into its running statistics twice. The recomputation
here runs with the norms' momentum at zero, so a checkpointed block updates its statistics once
per step, as an uncheckpointed one does.
"""

from contextlib import contextmanager

import torch.nn as nn
from torch.utils.checkpoint import checkpoint

__all__ = ["checkpoint_module"]


@contextmanager
def _frozen_batchnorm_stats(module: nn.Module):
    norms = [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [(m.momentum, m.num_batches_tracked.clone()) for m in norms]
    for m in norms:
        m.momentum = 0.0  # running = (1 - 0) * running + 0 * batch: unchanged
    try:
        yield
    finally:
        for m, (momentum, num_batches_tracked) in zip(norms, saved):
            m.momentum = momentum
            m.num_batches_tracked.copy_(num_batches_tracked)


def checkpoint_module(module: nn.Module, *args, fn=None):
    """
    ``module(*args)`` without keeping its intermediate activations: they are recomputed in the
    backward pass, with the module's BatchNorm statistics left as the forward pass set them.
    ``fn(*args)``, when given, is what runs in place of ``module(*args)``; it is meant for glue
    around the module (a concat of its inputs, say) whose result then need not be kept either.
    """
    fn = module if fn is None else fn
    first_pass = True

    def run(*args):
        nonlocal first_pass
        if first_pass:
            first_pass = False
            return fn(*args)
        with _frozen_batchnorm_stats(module):
            return fn(*args)

    return checkpoint(run, *args, use_reentrant=False)
