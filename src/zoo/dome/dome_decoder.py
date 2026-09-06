"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

The Dome decoder: ``DFINETransformer`` with Progressive Adaptive Query Initialization (PAQI) in
place of the fixed top-k query selection. Each image gets a core set of the ``min_num_select``
best encoder tokens plus, from the next ``max_num_select - min_num_select``, those that fall in
a window the density map marks as populated and survive a class-wise NMS whose IoU threshold
rises with the local density. Images in a batch therefore have different query counts. Without a
density map (the plain ``HybridEncoder``) every candidate is kept, i.e. a fixed
``max_num_select`` queries.
"""

import torch
import torch.nn.functional as F  # noqa: N812

from ...core import register
from ...misc.box_ops import box_cxcywh_to_xyxy
from .dfine_decoder import DFINETransformer
from .dynamic_nms import dynamic_nms

__all__ = ["DomeTransformer"]


@register()
class DomeTransformer(DFINETransformer):
    """
    Args (on top of ``DFINETransformer``'s; ``num_queries`` is replaced by ``max_num_select``):
        min_num_select / max_num_select: PAQI's core query count and the pool the adaptive
            queries are drawn from.
        nms_iou_low / nms_iou_high: the dynamic NMS threshold runs from ``low`` where the density
            map is 0 to ``high`` where it is 1 (the paper's IoU_N and IoU_M).
    """

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
        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            num_levels=num_levels,
            num_points=num_points,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            eval_spatial_size=eval_spatial_size,
            eval_idx=eval_idx,
            eps=eps,
            aux_loss=aux_loss,
            cross_attn_method=cross_attn_method,
            query_select_method=query_select_method,
            reg_max=reg_max,
            reg_scale=reg_scale,
            layer_scale=layer_scale,
            num_queries=max_num_select,
        )
        self.min_num_select = min_num_select
        self.max_num_select = max_num_select
        self.nms_iou_low = nms_iou_low
        self.nms_iou_high = nms_iou_high

    # ------------------------------------------------------------------ PAQI: query initialization

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

    def _get_decoder_input(self, memory, spatial_shapes, encoder_out):
        """
        PAQI. Returns the initial query contents and boxes (as logits, both detached and padded
        to the largest query count in the batch), the encoder-side predictions for the auxiliary
        loss, and the real query count of every image.
        """
        defe = encoder_out.get("defe")
        if defe is not None:
            # the criterion reads the query budget from here
            defe["min_num_select"] = self.min_num_select
            defe["max_num_select"] = self.max_num_select
        defe_window_mask = defe["defe_window_mask"] if defe is not None else None
        defe_feature = defe["density_map_pooled"] if defe is not None else None

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
