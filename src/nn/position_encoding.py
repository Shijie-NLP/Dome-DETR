"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch

__all__ = ["build_2d_sincos_position_embedding"]


def build_2d_sincos_position_embedding(w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0):
    """
    ``[1, w*h, embed_dim]`` sin/cos position embedding of a ``w`` x ``h`` grid, flattened
    column-major (all rows of column 0 first). The channel layout is
    ``[sin(x), cos(x), sin(y), cos(y)]`` with ``embed_dim // 4`` frequencies each; this is
    D-FINE's layout and the released checkpoints depend on it.
    """
    grid_w = torch.arange(int(w), dtype=torch.float32)
    grid_h = torch.arange(int(h), dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
    assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)

    out_w = grid_w.flatten()[..., None] @ omega[None]
    out_h = grid_h.flatten()[..., None] @ omega[None]

    return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]
