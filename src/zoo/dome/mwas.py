"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

MWAS, Masked Window Attention Sparsification: the stride-8 feature map is cut into
``window_size`` x ``window_size`` windows, and only the windows the DeFE density map marks as
populated go through a small transformer whose output is added back in place. Attention runs
inside each window and, alternately, across the selected windows at the same position
(``AxisPermutedEncoder``), so the sparse windows still exchange information.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ...misc.visualizer import dump_feature_map
from ...nn.transformer import TransformerEncoderLayer

__all__ = ["AxisPermutedEncoder", "MaskedWindowAttention"]


class AxisPermutedEncoder(nn.Module):
    """
    A stack of ``TransformerEncoderLayer`` applied to ``[N, L, C]`` tokens (N windows of L
    positions) twice per layer: once over the L positions of each window, then, with the first
    two axes swapped, over the N windows at each position. The positional embedding is the sum
    of a per-window absolute one (``glob_pos_embeds``, ``[N, C, h, w]``) and a shared relative one
    (``pos_embed``, ``[C, h, w]``).
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None, glob_pos_embeds=None) -> torch.Tensor:
        output = src
        n, c = glob_pos_embeds.shape[:2]
        pos = glob_pos_embeds.reshape(n, c, -1).permute(0, 2, 1) + pos_embed.reshape(c, -1).permute(1, 0).unsqueeze(0)
        pos_t = pos.permute(1, 0, 2).contiguous()
        for layer in self.layers:
            q = k = output + pos
            output = layer.attend(q, k, output, src_mask)
            # across windows: swap the window and position axes
            output = output.permute(1, 0, 2).contiguous()
            q = k = output + pos_t
            output = layer.attend(q, k, output, src_mask).permute(1, 0, 2).contiguous()

        if self.norm is not None:
            output = self.norm(output)
        return output


class MaskedWindowAttention(nn.Module):
    """
    Window attention over the populated windows of a feature map (see the module docstring).

    ``forward(features, mask, window_size, glob_pos_embed)`` takes the ``[B, C, H, W]`` features,
    a ``[B, 1, H, W]`` binary density mask, the window side in cells, and a ``[C, H, W]`` absolute
    position embedding. A window is selected when any mask cell inside it is set. Returns the
    features with the encoded windows added back, and the ``[B, H/ws, W/ws]`` window mask.
    """

    def __init__(self, embed_dim=256, num_heads=8, dim_feedforward=1024, num_layers=1, dropout=0.0, activation="relu"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # relative position inside a window, from its normalized (x, y)
        self.rel_pos_encoder = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, embed_dim))

        encoder_layer = TransformerEncoderLayer(
            embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )
        self.window_encoder = AxisPermutedEncoder(encoder_layer, self.num_layers)

    def forward(self, features, mask, window_size, glob_pos_embed):
        b, c, h, w = features.shape
        assert h % window_size == 0 and w % window_size == 0, "H and W must be divisible by window_size"
        ws = window_size
        rel_pos_embed = self._relative_embedding(ws)

        windows = self._as_windows(features, ws)  # [B, nh, nw, C, ws, ws]
        pos_windows = self._as_windows(glob_pos_embed[None], ws)[0]  # [nh, nw, C, ws, ws]
        window_mask = self._window_mask(mask, features.shape[-2:], ws)  # [B, nh, nw]
        dump_feature_map("defe_window_mask", window_mask.float().unsqueeze(1))

        out = features.clone()
        for i in range(b):
            rows, cols = torch.nonzero(window_mask[i], as_tuple=True)
            if rows.numel() == 0:
                raise RuntimeError(f"image {i}: the density mask selects no window")
            encoded = self._encode(windows[i][rows, cols], rel_pos_embed, pos_windows[rows, cols])
            # add the encoded windows back at their places
            delta = torch.zeros_like(windows[i])
            delta[rows, cols] = encoded
            out[i] += delta.permute(2, 0, 3, 1, 4).reshape(c, h, w)
        return out, window_mask

    @staticmethod
    def _as_windows(x, ws):
        """``[B, C, H, W]`` -> ``[B, H/ws, W/ws, C, ws, ws]``."""
        b, c, h, w = x.shape
        return x.view(b, c, h // ws, ws, w // ws, ws).permute(0, 2, 4, 1, 3, 5)

    @staticmethod
    def _window_mask(mask, feature_size, ws):
        """Which windows hold at least one set mask cell; the mask may be finer than the features."""
        h, w = feature_size
        kernel = (mask.shape[-2] // h * ws, mask.shape[-1] // w * ws)
        return F.max_pool2d(mask.float(), kernel_size=kernel, stride=kernel).squeeze(1) > 0

    def _encode(self, windows, rel_pos_embed, glob_pos_embeds):
        """Run the window encoder on ``[N, C, ws, ws]`` windows; same shape out."""
        n, c, h, w = windows.shape
        tokens = windows.reshape(n, c, -1).permute(0, 2, 1)
        tokens = self.window_encoder(tokens, pos_embed=rel_pos_embed, glob_pos_embeds=glob_pos_embeds)
        return tokens.permute(0, 2, 1).reshape(n, c, h, w)

    def _relative_embedding(self, ws):
        """``[C, ws, ws]`` embedding of each cell's normalized (x, y) inside the window."""
        device = self.rel_pos_encoder[0].weight.device
        grid_y, grid_x = torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")
        coords = torch.stack([grid_x / (ws - 1), grid_y / (ws - 1)], dim=-1).to(device)
        return self.rel_pos_encoder(coords).permute(2, 0, 1)
