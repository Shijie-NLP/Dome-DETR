"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

The Dome encoder: ``HybridEncoder`` with the two Dome additions in its ``enhance`` hook.

- DeFE, a light density head on the stride-4 level, predicts a per-pixel object density map and a
  per-image count. The criterion supervises the map with a Gaussian heatmap drawn from the boxes;
  the decoder sizes its query budget from the pooled map.
- MWAS runs a window attention over the stride-8 level restricted to the windows the density map
  marks as populated, and hands the decoder that window mask.
"""

import torch
import torch.nn.functional as F  # noqa: N812

from ...core import register
from ...misc.visualizer import SAVE_INTERMEDIATE_VISUALIZE_RESULT, dump_feature_map
from ...nn.position_encoding import build_2d_sincos_position_embedding
from .defe import LiteDeFE, adaptive_defe_filter, render_density_map
from .hybrid_encoder import HybridEncoder
from .mwas import MaskedWindowAttention

__all__ = ["DomeHybridEncoder"]


@register()
class DomeHybridEncoder(HybridEncoder):
    """
    ``HybridEncoder`` with DeFE on the stride-4 level and, optionally, MWAS on the stride-8 level.

    Args (on top of ``HybridEncoder``'s):
        defe_type: the density head; 'light' is the only one.
        use_mwas / mwas_window_size: window attention on the stride-8 level, restricted to the
            windows (of ``mwas_window_size`` x ``mwas_window_size`` stride-8 cells) the density
            map marks.

    The output dict gains ``defe`` with the density map (``defe_feature``), its per-window max
    (``density_map_pooled``), the count value (``reg_value``), the MWAS window mask
    (``defe_window_mask``, with MWAS) and, in training, the target heatmap (``gt_density_map``).
    """

    use_defe = True

    def __init__(
        self,
        in_channels=(512, 1024, 2048),
        feat_strides=(8, 16, 32),
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=(2,),
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=None,
        use_hybrid=True,
        defe_type="light",
        use_mwas=True,
        mwas_window_size=10,
    ):
        super().__init__(
            in_channels=in_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enc_act=enc_act,
            use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers,
            pe_temperature=pe_temperature,
            expansion=expansion,
            depth_mult=depth_mult,
            act=act,
            eval_spatial_size=eval_spatial_size,
            use_hybrid=use_hybrid,
        )
        self.defe_type = defe_type
        self.use_mwas = use_mwas
        self.mwas_window_size = mwas_window_size

        # MWAS is created before DeFE so the state-dict order matches the released checkpoints
        if self.use_mwas:
            self.mwas_processor = MaskedWindowAttention(
                embed_dim=hidden_dim, dim_feedforward=dim_feedforward, num_layers=1
            )
        if self.defe_type != "light":
            raise ValueError(f"Invalid defe_type: {self.defe_type}")
        self.DeFE = LiteDeFE()

    def enhance(self, proj_feats, img_inputs, targets):
        """Run DeFE (and MWAS, in place on the stride-8 level); returns ``{"defe": ...}``."""
        defe_feature, reg_value = self.DeFE(proj_feats[0])
        # the window grid lives on the stride-8 level
        H, W = proj_feats[1].shape[2:]
        ws = self.mwas_window_size
        defe_feature_pooled = F.adaptive_max_pool2d(defe_feature, (H // ws, W // ws))
        defe = {"reg_value": reg_value, "defe_feature": defe_feature, "density_map_pooled": defe_feature_pooled}

        if self.use_mwas:
            defe_feature_filtered = adaptive_defe_filter(
                F.interpolate(defe_feature_pooled, size=(H, W), mode="bilinear", align_corners=True)
            ).float()
            glob_pos_embed = (
                build_2d_sincos_position_embedding(W, H, embed_dim=self.hidden_dim)
                .permute(0, 2, 1)
                .view(-1, H, W)
                .to(proj_feats[1].device)
            )
            proj_feats[1], defe["defe_window_mask"] = self.mwas_processor(
                proj_feats[1], defe_feature_filtered, ws, glob_pos_embed
            )
            dump_feature_map("encoder_output_0", proj_feats[0])
            dump_feature_map("defe_feature_filtered", defe_feature_filtered)
        dump_feature_map("defe_feature", defe_feature)
        dump_feature_map("defe_feature_pooled", defe_feature_pooled)

        # the target heatmap, drawn from the boxes at image resolution; only the criterion reads
        # it, so it is built for training (and for the visualization dump)
        defe["gt_density_map"] = []
        if targets is not None and (self.training or SAVE_INTERMEDIATE_VISUALIZE_RESULT):
            B, _, img_h, img_w = img_inputs.shape
            heatmaps = []
            for b in range(B):
                boxes = targets[b]["boxes"]
                if not self.training:
                    # validation targets are xyxy pixels; the generator wants normalized cxcywh
                    boxes = boxes / boxes.new_tensor([img_w, img_h, img_w, img_h])
                    boxes = torch.cat([(boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]], dim=1)
                heatmaps.append(render_density_map(boxes, (img_h, img_w)))
            defe["gt_density_map"] = torch.stack(heatmaps).to(img_inputs.device)
            dump_feature_map("heatmap_gt", defe["gt_density_map"])
        return {"defe": defe}
