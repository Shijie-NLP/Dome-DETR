"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

The hybrid encoder of D-FINE (intra-scale attention on the top level, a CSP-ELAN feature pyramid
across levels) with the two Dome additions on the stride-4 / stride-8 levels:

- DeFE, a light density head on the stride-4 features, predicts a per-pixel object density map
  and a per-image count value. The criterion supervises the map with a Gaussian heatmap drawn
  from the ground-truth boxes; the decoder uses the pooled map to size its query budget.
- MWAS runs a window attention over the stride-8 features restricted to the windows the density
  map marks as populated, and hands the decoder that window mask.
"""

import copy
import os
import random
from collections import OrderedDict
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ...core import register
from .defe import GaussHeatmapGenerator, LiteDeFE
from .get_roi_features import TransformerEncoder, TransformerEncoderLayer, WindowProcessor
from .utils import get_activation

SAVE_INTERMEDIATE_VISUALIZE_RESULT = os.environ.get("SAVE_INTERMEDIATE_VISUALIZE_RESULT", "False") == "True"

__all__ = ["HybridEncoder"]


def _visualize(name: str, feature: torch.Tensor):
    """Dump a [B, C, H, W] map under ``name`` when SAVE_INTERMEDIATE_VISUALIZE_RESULT is set."""
    if SAVE_INTERMEDIATE_VISUALIZE_RESULT:
        from tools.visualize_src_flatten import visualize_src_flatten

        visualize_src_flatten(feature.permute(0, 2, 3, 1), [tuple(feature.shape[2:4])], name, False)


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """The kernel and bias of the single conv equivalent to ``conv`` followed by ``bn``."""
    std = (bn.running_var + bn.eps).sqrt()
    scale = bn.weight / std
    return conv.weight * scale.reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * scale


class ConvNormLayer_fuse(nn.Module):  # noqa: N801
    """Conv -> BN -> act, foldable into a single conv for deployment."""

    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = (
            ch_in,
            ch_out,
            kernel_size,
            stride,
            g,
            padding,
            bias,
        )

    def forward(self, x):
        if hasattr(self, "conv_bn_fused"):
            return self.act(self.conv_bn_fused(x))
        return self.act(self.norm(self.conv(x)))

    def convert_to_deploy(self):
        if not hasattr(self, "conv_bn_fused"):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in, self.ch_out, self.kernel_size, self.stride, groups=self.g, padding=self.padding, bias=True
            )
        kernel, bias = _fuse_conv_bn(self.conv, self.norm)
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__("conv")
        self.__delattr__("norm")


class ConvNormLayer(nn.Module):
    """Conv -> BN -> act."""

    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SCDown(nn.Module):
    """Spatial-channel decoupled downsampling: a 1x1 pointwise conv then a strided depthwise conv."""

    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    """RepVGG block: parallel 3x3 and 1x1 conv-BN branches, foldable into one 3x3 conv for deployment."""

    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else act

    def forward(self, x):
        if hasattr(self, "conv"):
            return self.act(self.conv(x))
        return self.act(self.conv1(x) + self.conv2(x))

    def convert_to_deploy(self):
        if not hasattr(self, "conv"):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__("conv1")
        self.__delattr__("conv2")

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = _fuse_conv_bn(self.conv1.conv, self.conv1.norm)
        kernel1x1, bias1x1 = _fuse_conv_bn(self.conv2.conv, self.conv2.norm)
        return kernel3x3 + F.pad(kernel1x1, [1, 1, 1, 1]), bias3x3 + bias1x1


class CSPLayer(nn.Module):
    """Cross-stage partial layer: a stack of bottlenecks on one 1x1 branch, summed with a plain 1x1 branch."""

    def __init__(
        self, in_channels, out_channels, num_blocks=3, expansion=1.0, bias=False, act="silu", bottletype=VGGBlock
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(
            *[bottletype(hidden_channels, hidden_channels, act=get_activation(act)) for _ in range(num_blocks)]
        )
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


class RepNCSPELAN4(nn.Module):
    """The GELAN fusion block (from YOLOv9): split, two chained CSP stages, concat everything, 1x1 out."""

    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


@register()
class HybridEncoder(nn.Module):
    """
    Args:
        in_channels / feat_strides: the backbone levels, lowest stride first.
        hidden_dim: every level is projected to this width.
        use_encoder_idx / num_encoder_layers / nhead / dim_feedforward / dropout / enc_act /
            pe_temperature: the intra-scale transformer applied to the listed levels.
        expansion / depth_mult / act: width, depth and activation of the pyramid's fusion blocks.
        use_hybrid: run the top-down / bottom-up pyramid; off, the projected levels are returned.
        eval_spatial_size: (h, w) at evaluation, used to precompute the position embeddings.
        use_defe / defe_type: the density head on the stride-4 level ('light' is the only type).
        use_mwas / mwas_window_size: window attention on the stride-8 level, restricted to the
            windows (of ``mwas_window_size`` x ``mwas_window_size`` stride-8 cells) the density
            map marks; needs ``use_defe``.

    ``forward`` returns a dict: ``feats`` (the pyramid), ``img_inputs`` (the image, for the
    decoder's window geometry) and, with DeFE, ``defe`` with the density map (``defe_feature``),
    its per-window max (``density_map_pooled``), the count value (``reg_value``), the MWAS window
    mask (``defe_window_mask``) and, in training, the target heatmap (``gt_density_map``).
    """

    __share__ = ["eval_spatial_size"]

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
        use_defe=False,
        defe_type="default",
        use_mwas=False,
        mwas_window_size=20,
    ):
        super().__init__()
        if use_mwas and not use_defe:
            raise ValueError("use_mwas needs use_defe: the window mask comes from the density map")
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.pos_embeds = []
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.use_hybrid = use_hybrid
        self.use_defe = use_defe
        self.defe_type = defe_type
        self.use_mwas = use_mwas
        self.mwas_window_size = mwas_window_size

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("conv", nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                            ("norm", nn.BatchNorm2d(hidden_dim)),
                        ]
                    )
                )
            )

        # intra-scale transformer, one per listed level
        if self.num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(
                hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=enc_act
            )
            self.encoder = nn.ModuleList(
                [
                    TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
                    for _ in range(len(use_encoder_idx))
                ]
            )

        if self.use_hybrid:
            fusion = dict(c3=hidden_dim * 2, c4=round(expansion * hidden_dim // 2), n=round(3 * depth_mult), act=act)
            # top-down: lateral 1x1 on the coarser level, upsample, fuse with the finer one
            self.lateral_convs = nn.ModuleList()
            self.fpn_blocks = nn.ModuleList()
            for _ in range(len(in_channels) - 1):
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
                self.fpn_blocks.append(RepNCSPELAN4(hidden_dim * 2, hidden_dim, **fusion))
            # bottom-up: downsample the finer level, fuse with the coarser one
            self.downsample_convs = nn.ModuleList()
            self.pan_blocks = nn.ModuleList()
            for _ in range(len(in_channels) - 1):
                self.downsample_convs.append(nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2)))
                self.pan_blocks.append(RepNCSPELAN4(hidden_dim * 2, hidden_dim, **fusion))

        if self.use_defe:
            # MWAS is created before DeFE so the state-dict order matches the released checkpoints
            if self.use_mwas:
                self.mwas_processor = WindowProcessor(
                    embed_dim=hidden_dim, dim_feedforward=dim_feedforward, num_layers=1
                )
            if self.defe_type != "light":
                raise ValueError(f"Invalid defe_type: {self.defe_type}")
            self.DeFE = LiteDeFE()

        self._build_eval_pos_embeds()

    def _build_eval_pos_embeds(self):
        """Position embeddings for the evaluation size, one per intra-scale level, computed once."""
        if not self.eval_spatial_size:
            return
        for idx in self.use_encoder_idx:
            stride = self.feat_strides[idx]
            self.pos_embeds.append(
                self.build_2d_sincos_position_embedding(
                    ceil(self.eval_spatial_size[1] / stride),
                    ceil(self.eval_spatial_size[0] / stride),
                    self.hidden_dim,
                    self.pe_temperature,
                )
            )

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """[1, w*h, embed_dim] sin/cos embedding of a w x h grid (D-FINE's layout, kept for its checkpoints)."""
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

    @staticmethod
    def adaptive_defe_filter(defe_feature, init_thresh=0.05, step=0.01):
        """
        Binarize a density map [B, 1, H, W] per image, lowering the threshold from ``init_thresh``
        in steps of ``step`` until something passes. An all-zero map gets one random cell so that
        the window attention always has a window to work on.
        """
        final_mask = torch.zeros_like(defe_feature, dtype=torch.bool)
        for b in range(defe_feature.shape[0]):
            single_feat = defe_feature[b : b + 1]
            current_thresh = init_thresh
            while current_thresh >= 0:
                mask = single_feat > current_thresh
                if mask.any():
                    final_mask[b : b + 1] = mask
                    break
                current_thresh = round(current_thresh - step, 2)
            else:
                final_mask[
                    b, :, random.randint(0, single_feat.shape[2] - 1), random.randint(0, single_feat.shape[3] - 1)
                ] = True
                print(f"Batch {b}: No valid region found, use random point enhancement")
        return final_mask

    def _defe(self, proj_feats, img_inputs, targets):
        """Run DeFE (and MWAS, in place on the stride-8 level); returns the ``defe`` output dict."""
        defe_feature, reg_value = self.DeFE(proj_feats[0])
        # the window grid lives on the stride-8 level
        H, W = proj_feats[1].shape[2:]
        ws = self.mwas_window_size
        defe_feature_pooled = F.adaptive_max_pool2d(defe_feature, (H // ws, W // ws))
        defe = {"reg_value": reg_value, "defe_feature": defe_feature, "density_map_pooled": defe_feature_pooled}

        if self.use_mwas:
            defe_feature_filtered = self.adaptive_defe_filter(
                F.interpolate(defe_feature_pooled, size=(H, W), mode="bilinear", align_corners=True)
            ).float()
            glob_pos_embed = (
                self.build_2d_sincos_position_embedding(W, H, embed_dim=self.hidden_dim)
                .permute(0, 2, 1)
                .view(-1, H, W)
                .to(proj_feats[1].device)
            )
            proj_feats[1], defe["defe_window_mask"] = self.mwas_processor(
                proj_feats[1], defe_feature_filtered, ws, glob_pos_embed
            )
            _visualize("encoder_output_0", proj_feats[0])
            _visualize("defe_feature_filtered", defe_feature_filtered)
        _visualize("defe_feature", defe_feature)
        _visualize("defe_feature_pooled", defe_feature_pooled)

        # the target heatmap, drawn from the boxes at image resolution; only the criterion reads
        # it, so it is built for training (and for the visualization dump)
        defe["gt_density_map"] = []
        if targets is not None and (self.training or SAVE_INTERMEDIATE_VISUALIZE_RESULT):
            B, _, img_h, img_w = img_inputs.shape
            heatmap_generator = GaussHeatmapGenerator(img_size=(img_h, img_w))
            heatmaps = []
            for b in range(B):
                boxes = targets[b]["boxes"]
                if not self.training:
                    # validation targets are xyxy pixels; the generator wants normalized cxcywh
                    boxes = boxes / boxes.new_tensor([img_w, img_h, img_w, img_h])
                    boxes = torch.cat([(boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]], dim=1)
                heatmaps.append(heatmap_generator(boxes))
            defe["gt_density_map"] = torch.stack(heatmaps).to(img_inputs.device)
            _visualize("heatmap_gt", defe["gt_density_map"])
        return defe

    def forward(self, feats, img_inputs, targets=None):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        _visualize("backbone_output_0", proj_feats[0])

        out = {"img_inputs": img_inputs}
        if self.use_defe:
            out["defe"] = self._defe(proj_feats, img_inputs, targets)

        # intra-scale transformer
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)  # [B, HW, C]
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature)
                else:
                    pos_embed = self.pos_embeds[i]
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed.to(src_flatten.device))
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        if not self.use_hybrid:
            out["feats"] = proj_feats
            return out

        # top-down: coarsest level first, each finer level fused with the upsampled result
        inner_outs = [proj_feats[-1]]
        for i, idx in enumerate(range(len(self.in_channels) - 1, 0, -1)):
            feat_high = self.lateral_convs[i](inner_outs[0])
            feat_low = proj_feats[idx - 1]
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, size=feat_low.shape[2:], mode="bilinear", align_corners=True)
            inner_outs.insert(0, self.fpn_blocks[i](torch.concat([upsample_feat, feat_low], dim=1)))

        # bottom-up: finest level first, each coarser level fused with the downsampled result
        outs = [inner_outs[0]]
        for i in range(len(self.in_channels) - 1):
            downsample_feat = self.downsample_convs[i](outs[-1])
            outs.append(self.pan_blocks[i](torch.concat([downsample_feat, inner_outs[i + 1]], dim=1)))

        out["feats"] = outs
        return out
