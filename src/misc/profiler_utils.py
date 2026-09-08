"""
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
"""

import copy

import torch
from torch.utils.flop_counter import FlopCounterMode


def stats(cfg, input_shape: tuple = (1, 3, 640, 640)) -> tuple[int, dict]:
    """
    Parameter count and FLOPs / MACs of the config's model in deploy form, at the training
    ``base_size`` of the collate function (``input_shape`` when the config has none), counted
    by torch's FlopCounterMode on one CPU forward (matmuls, convolutions and attention; the
    sampling ops it leaves out are a fraction of a percent). Returns the count and a one-line
    summary for the log.
    """
    base_size = cfg.train_dataloader.collate_fn.base_size
    if isinstance(base_size, (list, tuple)):
        input_shape = (1, 3, base_size[0], base_size[1])
    else:
        input_shape = (1, 3, base_size, base_size)

    model_for_info = copy.deepcopy(cfg.model).deploy()  # on the model's device: a CPU forward at 800x800 takes seconds
    device = next(model_for_info.parameters()).device
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model_for_info(torch.zeros(input_shape, device=device))
    flops = counter.get_total_flops()
    params = sum(p.numel() for p in model_for_info.parameters())
    return params, {f"Model FLOPs:{flops / 1e9:.4g} GFLOPs   MACs:{flops / 2e9:.4g} GMACs   Params:{params}"}
