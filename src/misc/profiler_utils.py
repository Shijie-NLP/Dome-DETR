"""
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
"""

import copy


def stats(cfg, input_shape: tuple = (1, 3, 640, 640)) -> tuple[int, dict]:
    """
    Parameter count and FLOPs / MACs of the config's model in deploy form, at the training
    ``base_size`` of the collate function (``input_shape`` when the config has none). Returns the
    count and a one-line summary for the log.
    """
    # imported here: calflops pulls in transformers and scikit-learn, 2.5 s that every dataloader
    # worker would otherwise pay on start-up (importing the dataset module imports the package)
    from calflops import calculate_flops

    base_size = cfg.train_dataloader.collate_fn.base_size
    if isinstance(base_size, (list, tuple)):
        input_shape = (1, 3, base_size[0], base_size[1])
    else:
        input_shape = (1, 3, base_size, base_size)

    model_for_info = copy.deepcopy(cfg.model).deploy()
    flops, macs, _ = calculate_flops(
        model=model_for_info,
        input_shape=input_shape,
        output_as_string=True,
        output_precision=4,
        print_detailed=False,
    )
    params = sum(p.numel() for p in model_for_info.parameters())
    return params, {f"Model FLOPs:{flops}   MACs:{macs}   Params:{params}"}
