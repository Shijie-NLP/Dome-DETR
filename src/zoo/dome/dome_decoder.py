"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

The D-FINE decoder with Dome's Progressive Adaptive Query Initialization (PAQI): instead of a
fixed number of queries, each image gets a core set of the ``min_num_select`` best encoder
tokens plus, from the next ``max_num_select - min_num_select``, those that fall in a window the
density map marks as populated and survive a class-wise NMS whose IoU threshold rises with the
local density. Images in a batch therefore have different query counts and are padded to the
largest; ``batch_queries_num`` tells the criterion and the denoising mask how many are real.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.nn.init as init

from ...core import register
from ...misc.box_ops import box_cxcywh_to_xyxy
from ...misc.visualizer import SAVE_INTERMEDIATE_VISUALIZE_RESULT, dump_boxes
from ...nn.functional import bias_init_with_prob, inverse_sigmoid
from ...nn.transformer import MLP, TransformerDecoderLayer
from .denoising import get_contrastive_denoising_training_group
from .dynamic_nms import dynamic_nms
from .fdr import LQE, Integral, distance2bbox, weighting_function

__all__ = ["DomeTransformer"]


class TransformerDecoder(nn.Module):
    """
    The decoder stack with Fine-grained Distribution Refinement: every layer predicts a
    correction to the binned edge distributions of the layer before, the boxes are decoded from
    the accumulated distribution around the first layer's reference boxes, and a location
    quality estimator adjusts the class scores. Layers past ``eval_idx`` (used in training only)
    can be ``layer_scale`` times wider.
    """

    def __init__(
        self,
        hidden_dim,
        decoder_layer,
        decoder_layer_wide,
        num_layers,
        num_head,
        reg_max,
        reg_scale,
        up,
        eval_idx=-1,
        layer_scale=2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max)) for _ in range(num_layers)])

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        """The encoder memory as per-level, per-head values for the deformable cross-attention."""
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[: self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]])

    def forward(
        self,
        target,
        ref_points_unact,
        memory,
        spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
        pre_bbox_head,
        integral,
        up,
        reg_scale,
        attn_mask=None,
        memory_mask=None,
        img_input=None,
    ):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes, dec_out_logits, dec_out_pred_corners, dec_out_refs = [], [], [], []
        project = self.project if hasattr(self, "project") else weighting_function(self.reg_max, up, reg_scale)

        ref_points_detach = F.sigmoid(ref_points_unact)
        if SAVE_INTERMEDIATE_VISUALIZE_RESULT:
            dump_boxes("ref_bbox", img_input, ref_points_detach[0])

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            # the wider training-only layers work on interpolated queries and values
            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

            if i == 0:
                # the first layer predicts plain boxes; they anchor the distributions of every layer
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            # refine the edge distributions, carrying the previous layer's correction along
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if self.training or i == self.eval_idx:
                scores = self.lqe_layers[i](score_head[i](output), pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        if SAVE_INTERMEDIATE_VISUALIZE_RESULT and dec_out_bboxes:
            probs = dec_out_logits[-1][0].softmax(-1)
            dump_boxes("dec_out_bboxes", img_input, dec_out_bboxes[-1][0], probs.argmax(-1), probs.max(-1).values)

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
        )


@register()
class DomeTransformer(nn.Module):
    """
    Args:
        feat_channels / feat_strides / num_levels: the encoder levels; levels beyond those given
            are made by strided 3x3 convs on the last one.
        num_layers / eval_idx: decoder depth, and the layer whose output is used at inference
            (the later ones train the earlier ones and are dropped by ``convert_to_deploy``).
        num_denoising / label_noise_ratio / box_noise_scale: contrastive denoising queries.
        reg_max / reg_scale: the FDR edge distributions.
        query_select_method: how encoder tokens are ranked, ``default`` (best class score),
            ``one2many`` (every class score) or ``agnostic`` (a single objectness score).
        min_num_select / max_num_select: PAQI's core query count and the pool the adaptive
            queries are drawn from.
        nms_iou_low / nms_iou_high: the dynamic NMS threshold runs from ``low`` where the density
            map is 0 to ``high`` where it is 1 (the paper's IoU_N and IoU_M).
    """

    __share__ = ["num_classes", "eval_spatial_size"]

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        feat_channels=(512, 1024, 2048),
        feat_strides=(8, 16, 32, 64, 128),
        num_levels=5,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        min_num_select=300,
        max_num_select=1500,
        nms_iou_low=0.4,
        nms_iou_high=0.9,
    ):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        assert query_select_method in ("default", "one2many", "agnostic"), query_select_method
        assert cross_attn_method in ("default", "discrete"), cross_attn_method

        feat_strides = list(feat_strides)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max
        self.min_num_select = min_num_select
        self.max_num_select = max_num_select
        self.nms_iou_low = nms_iou_low
        self.nms_iou_high = nms_iou_high
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # transformer
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        layer_args = dict(
            d_model=hidden_dim,
            n_head=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=num_levels,
            n_points=num_points,
            cross_attn_method=cross_attn_method,
        )
        decoder_layer = TransformerDecoderLayer(**layer_args)
        decoder_layer_wide = TransformerDecoderLayer(**layer_args, layer_scale=layer_scale)
        self.decoder = TransformerDecoder(
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            nhead,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
        )

        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)

        # encoder output: token scores and boxes for query selection
        self.enc_output = nn.Sequential(
            OrderedDict([("proj", nn.Linear(hidden_dim, hidden_dim)), ("norm", nn.LayerNorm(hidden_dim))])
        )
        self.enc_score_head = nn.Linear(hidden_dim, 1 if query_select_method == "agnostic" else num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        # decoder heads, one per layer; the training-only layers past eval_idx may be wider
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        num_wide = num_layers - self.eval_idx - 1
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)]
            + [nn.Linear(scaled_dim, num_classes) for _ in range(num_wide)]
        )
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3) for _ in range(self.eval_idx + 1)]
            + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3) for _ in range(num_wide)]
        )
        self.integral = Integral(self.reg_max)

        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]])
        self.dec_bbox_head = nn.ModuleList(
            [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
        )

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, "layers"):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        """A conv-BN projection to ``hidden_dim`` per level; extra levels downsample the last one with stride 2."""

        def proj(in_channels, kernel_size, stride):
            if in_channels == self.hidden_dim and kernel_size == 1:
                return nn.Identity()
            conv = nn.Conv2d(in_channels, self.hidden_dim, kernel_size, stride, padding=kernel_size // 2, bias=False)
            return nn.Sequential(OrderedDict([("conv", conv), ("norm", nn.BatchNorm2d(self.hidden_dim))]))

        self.input_proj = nn.ModuleList(proj(c, 1, 1) for c in feat_channels)
        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(proj(in_channels, 3, 2))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: list[torch.Tensor]):
        """The projected levels, and the same flattened to ``[b, sum(h*w), c]`` with their ``(h, w)``."""
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        for i in range(len(feats), self.num_levels):
            source = feats[-1] if i == len(feats) else proj_feats[-1]
            proj_feats.append(self.input_proj[i](source))

        spatial_shapes = [list(feat.shape[2:]) for feat in proj_feats]
        memory = torch.concat([feat.flatten(2).permute(0, 2, 1) for feat in proj_feats], 1)
        return proj_feats, memory, spatial_shapes

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"):
        """One anchor box (as logits) per token of every level; boxes too close to the border are marked invalid."""
        if spatial_shapes is None:
            eval_h, eval_w = self.eval_spatial_size
            spatial_shapes = [[int(eval_h / s), int(eval_w / s)] for s in self.feat_strides]

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    # ------------------------------------------------------------------ PAQI: query initialization

    def _select_topk(self, memory, outputs_logits, outputs_anchors_unact, topk):
        """The ``topk`` tokens by ``query_select_method``, with their logits and anchors."""
        if self.query_select_method == "default":
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
        else:  # agnostic
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        def gather(x):
            return x.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        return gather(memory), gather(outputs_logits), gather(outputs_anchors_unact)

    @staticmethod
    def _in_marked_windows(anchors_unact, window_mask):
        """Which anchors (logit cxcywh) have their centre in a window ``window_mask`` ``[B, rows, cols]`` marks."""
        b = anchors_unact.shape[0]
        n_rows, n_cols = window_mask.shape[1], window_mask.shape[2]
        cx, cy = F.sigmoid(anchors_unact[..., 0]), F.sigmoid(anchors_unact[..., 1])
        col = (cx * n_cols).long().clamp(0, n_cols - 1)
        row = (cy * n_rows).long().clamp(0, n_rows - 1)
        return window_mask[torch.arange(b, device=anchors_unact.device).view(-1, 1), row, col]

    def _density_nms(self, boxes_cxcywh, logits, density_map):
        """
        Class-wise NMS over one image's candidate boxes (normalized cxcywh) with an IoU threshold
        that rises from ``nms_iou_low`` to ``nms_iou_high`` with the density under each box's
        centre (``density_map`` is ``[1, h, w]``). Returns the indices to keep.
        """
        cx, cy = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1]
        h, w = density_map.shape[1:]
        row = (cy * (h - 1)).long().clamp(0, h - 1)
        col = (cx * (w - 1)).long().clamp(0, w - 1)
        density = density_map[:, row, col].squeeze(0).detach()
        iou_thresholds = self.nms_iou_low + (self.nms_iou_high - self.nms_iou_low) * density
        scores, class_ids = logits.max(dim=1)
        return dynamic_nms(box_cxcywh_to_xyxy(boxes_cxcywh), scores, class_ids, iou_thresholds)

    def _get_decoder_input(self, memory, spatial_shapes, defe_window_mask=None, defe_feature=None):
        """
        PAQI. Returns the initial query contents and boxes (as logits, both detached and padded
        to the largest query count in the batch), the encoder-side predictions for the auxiliary
        loss, and the real query count of every image.
        """
        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)
        memory = valid_mask.to(memory.dtype) * memory

        output_memory: torch.Tensor = self.enc_output(memory)
        enc_outputs_logits: torch.Tensor = self.enc_score_head(output_memory)

        topk_memory, topk_logits, topk_anchors = self._select_topk(
            output_memory, enc_outputs_logits, anchors, self.max_num_select
        )
        b = topk_anchors.size(0)
        min_num = self.min_num_select

        # the core queries are kept as they are; the rest must sit in a populated window
        if defe_window_mask is not None:
            selected_mask = self._in_marked_windows(topk_anchors[:, min_num:], defe_window_mask)
        else:
            selected_mask = torch.ones_like(topk_anchors[:, min_num:, 0], dtype=torch.bool)

        per_image = []  # (memory, logits, bbox_unact) per image, after window filtering and NMS
        for i in range(b):
            keep = torch.cat([torch.ones(min_num, dtype=torch.bool, device=memory.device), selected_mask[i]])
            mem = topk_memory[i][keep]
            logits = topk_logits[i][keep]
            bbox_unact = self.enc_bbox_head(mem) + topk_anchors[i][keep]

            if defe_feature is not None and logits.size(0) > 0:
                keep_idx = self._density_nms(F.sigmoid(bbox_unact), logits, defe_feature[i])
                # the core queries are never suppressed
                keep_idx = torch.cat([torch.arange(min_num, device=keep_idx.device), keep_idx[keep_idx >= min_num]])
                mem, logits, bbox_unact = mem[keep_idx], logits[keep_idx], bbox_unact[keep_idx]
            per_image.append((mem, logits, bbox_unact))

        batch_queries_num = [mem.size(0) for mem, _, _ in per_image]
        max_total = max(batch_queries_num)
        padded_memory = torch.zeros((b, max_total, topk_memory.size(-1)), device=memory.device)
        padded_logits = torch.zeros((b, max_total, topk_logits.size(-1)), device=memory.device)
        padded_bbox_unact = torch.zeros((b, max_total, 4), device=memory.device)
        for i, (mem, logits, bbox_unact) in enumerate(per_image):
            n = batch_queries_num[i]
            padded_memory[i, :n] = mem
            padded_logits[i, :n] = logits
            padded_bbox_unact[i, :n] = bbox_unact

        # the criterion masks the padded entries with batch_queries_num
        enc_topk_bboxes_list = [F.sigmoid(padded_bbox_unact)]
        enc_topk_logits_list = [padded_logits]
        return (
            padded_memory.detach(),
            padded_bbox_unact.detach(),
            enc_topk_bboxes_list,
            enc_topk_logits_list,
            batch_queries_num,
        )

    # ------------------------------------------------------------------ forward

    def forward(self, encoder_out, targets=None):
        feats = encoder_out["feats"]
        img_inputs = encoder_out["img_inputs"]

        defe = encoder_out.get("defe")
        if defe is not None:
            # the criterion reads the query budget from here
            defe["min_num_select"] = self.min_num_select
            defe["max_num_select"] = self.max_num_select
        defe_window_mask = defe["defe_window_mask"] if defe is not None else None
        defe_feature = defe["density_map_pooled"] if defe is not None else None

        _, memory, spatial_shapes = self._get_encoder_input(feats)

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list, batch_queries_num = (
            self._get_decoder_input(memory, spatial_shapes, defe_window_mask, defe_feature)
        )
        num_queries = max(batch_queries_num)

        # denoising queries are prepended to the matching queries during training
        dn_meta = None
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets,
                self.num_classes,
                num_queries,
                self.denoising_class_embed,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
                batch_queries_num=batch_queries_num,
                num_heads=self.nhead,
            )
            init_ref_points_unact = torch.concat([denoising_bbox_unact, init_ref_points_unact], dim=1)
            init_ref_contents = torch.concat([denoising_logits, init_ref_contents], dim=1)
        else:
            attn_mask = None

        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            img_input=img_inputs,
        )

        if dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta["dn_num_split"], dim=2)

        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}
        if self.training:
            out["pred_corners"] = out_corners[-1]
            out["ref_points"] = out_refs[-1]
            out["up"] = self.up
            out["reg_scale"] = self.reg_scale

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._layer_outputs(
                out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1], out_corners[-1], out_logits[-1]
            )
            out["enc_aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b} for a, b in zip(enc_topk_logits_list, enc_topk_bboxes_list)
            ]
            out["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}

            if dn_meta is not None:
                out["dn_outputs"] = self._layer_outputs(
                    dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs, dn_out_corners[-1], dn_out_logits[-1]
                )
                out["dn_pre_outputs"] = {"pred_logits": dn_pre_logits, "pred_boxes": dn_pre_bboxes}
                out["dn_meta"] = dn_meta

        for key, value in encoder_out.items():
            if key != "feats":
                out[key] = value
        out["batch_queries_num"] = batch_queries_num
        return out

    @staticmethod
    @torch.jit.unused
    def _layer_outputs(logits, boxes, corners, refs, teacher_corners=None, teacher_logits=None):
        """One prediction dict per decoder layer, for the auxiliary and denoising losses."""
        return [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_corners": c,
                "ref_points": d,
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,
            }
            for a, b, c, d in zip(logits, boxes, corners, refs)
        ]
