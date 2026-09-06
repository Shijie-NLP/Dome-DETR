"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.nn as nn

from ...core import register
from .dome_decoder import DomeTransformer
from .dome_encoder import DomeHybridEncoder

__all__ = ["DOME"]


@register()
class DOME(nn.Module):
    """
    Backbone -> encoder -> decoder, wired the Dome way.

    Unlike D-FINE, the encoder also receives the input image and the targets: DeFE builds its
    density supervision from the image size and the ground-truth boxes, and hands the decoder a
    dict (features plus ``img_inputs`` and ``defe``) rather than a plain feature list, which is
    what lets the decoder pick a per-image query budget from the predicted density.
    """

    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(self, backbone: nn.Module, encoder: nn.Module = None, decoder: nn.Module = None):
        super().__init__()
        if encoder is None or decoder is None:
            raise ValueError(
                "DOME needs an encoder and a decoder: set `DOME: {encoder: ..., decoder: ...}` in the model "
                "config (DomeHybridEncoder + DomeTransformer for Dome-DETR, HybridEncoder + DFINETransformer "
                "for the D-FINE baseline)"
            )
        if isinstance(decoder, DomeTransformer) and not isinstance(encoder, DomeHybridEncoder):
            raise ValueError(
                f"DomeTransformer needs the density map of DomeHybridEncoder, got {type(encoder).__name__}; "
                "for the D-FINE baseline use DFINETransformer"
            )
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    @property
    def wiring(self) -> str:
        return (
            f"{type(self).__name__}(backbone={type(self.backbone).__name__}, "
            f"encoder={type(self.encoder).__name__}, decoder={type(self.decoder).__name__})"
        )

    def forward(self, x, targets=None):
        feats = self.backbone(x)
        encoder_out = self.encoder(feats, x, targets)
        return self.decoder(encoder_out, targets)

    def deploy(self):
        """Switch to inference: eval mode, with every module that can fold its layers folded."""
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
