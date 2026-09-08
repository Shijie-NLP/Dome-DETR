"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

Ground-truth-forced query selection (working name ``MaxIoUTransformer``): a ``DFINETransformer``
whose query selection replaces the top-k by objectness, with no denoising queries. Everything
past the selection is D-FINE's: the encoder's predictions on the queries and the decoder's
outputs are Hungarian-matched by the criterion. The one difference in the losses is on the
encoder side, where the class logits are trained as plain 0/1 objectness (``loss_obj``, no
IoU-aware target, ``enc_losses`` of ``DomeCriterion``), so that a probability threshold on
them can serve as the selection rule.

- The objectness of a token is its highest class logit; a token passes when its sigmoid exceeds
  ``obj_threshold`` (0.5: the head's decision boundary, logit 0).
- Training (with ground truths in the batch): every ground truth forces the ``assign_k`` tokens
  whose predicted boxes match it best (``assign_metric``: ``gaussian``, ``nwd``, ``giou`` or
  ``iou``) among the ``(2 * assign_radius + 1)``-cell windows around its centre cell on every
  level (index arithmetic, no ``cdist``) into the queries. No assignment: a token several ground
  truths want is forced once, and the criterion's matching decides who predicts what. The rest
  of the queries are the unforced tokens that pass the objectness threshold; an image short of
  ``min_queries`` queries fills up with the best unpassed tokens by objectness, and one beyond
  ``num_queries`` keeps its forced tokens and the best passed ones up to it. Images differ in
  query count (padded to the largest, ``batch_queries_num`` tells the criterion).
- Inference: the tokens that pass the objectness threshold, clamped to ``[min_queries,
  num_queries]`` by the logit, so the query count is per image; ``num_queries`` is only a memory
  guard. As the head learns to pass the forced tokens, the training set converges to the
  inference set plus a vanishing forced part. ``infer_rule='topk'`` restores the plain top-k by
  objectness.
- ``last_assign_stats`` reports, per image, the ground truths, the forced tokens, how many of
  them the head already lets through (``selected``, its recall in training), the rule-selected
  queries and the forced tokens per level (``levels``: a drift of tiny objects' tokens towards
  coarse levels shows here).

Cost: the projection and the class head run on every token once, without a graph, for the
selection, and with the box head again on the queries alone with one; no full-token activation
is kept for the backward pass (D-FINE keeps the projection's and the class head's). Host syncs
per batch: the query counts.
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn.utils.rnn import pad_sequence

from ...core import register
from ...misc.box_ops import box_cxcywh_to_xyxy, gaussian_box_similarity
from .dfine_decoder import DecoderInput, DFINETransformer

__all__ = ["MaxIoUTransformer"]


def _pair_iou_giou(a_xyxy, b_xyxy):
    """IoU and GIoU of boxes at matching positions of two ``[..., 4]`` xyxy tensors (broadcastable)."""
    lt = torch.max(a_xyxy[..., :2], b_xyxy[..., :2])
    rb = torch.min(a_xyxy[..., 2:], b_xyxy[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a_xyxy[..., 2] - a_xyxy[..., 0]) * (a_xyxy[..., 3] - a_xyxy[..., 1])
    area_b = (b_xyxy[..., 2] - b_xyxy[..., 0]) * (b_xyxy[..., 3] - b_xyxy[..., 1])
    union = area_a + area_b - inter
    iou = inter / union
    lt = torch.min(a_xyxy[..., :2], b_xyxy[..., :2])
    rb = torch.max(a_xyxy[..., 2:], b_xyxy[..., 2:])
    wh = (rb - lt).clamp(min=0)
    enclosing = wh[..., 0] * wh[..., 1]
    return iou, iou - (enclosing - union) / enclosing


@register()
class MaxIoUTransformer(DFINETransformer):
    """
    Args (on top of ``DFINETransformer``'s; ``num_queries`` is the most queries an image gets):
        assign_metric: how a ground truth ranks the tokens' predicted boxes: ``gaussian`` (the
            boxes' Gaussian similarity, ``box_ops.gaussian_box_similarity``: scale-invariant,
            and a small box that just misses a tiny ground truth still beats a coarse box that
            merely contains it), ``nwd`` (the same ranking as the normalized Gaussian Wasserstein
            distance, i.e. by plain distance in ``(cx, cy, w/2, h/2)`` space: an offset counts in
            pixels whatever the box size), ``giou`` or ``iou`` (under both, any box containing
            the ground truth beats any box not overlapping it).
        assign_k: tokens each ground truth forces into the queries, its most similar ones.
        assign_radius: cells around the ground truth's centre cell, per level, that are candidates
            (1: a 3x3 window on each level).
        min_queries: the least queries an image gets, in training (the forced tokens and the
            passed ones, then the best unpassed tokens by objectness) and at inference alike.
        obj_threshold: the objectness probability (the highest class logit's sigmoid) a token
            passes at; 0.5 is the head's decision boundary, logit 0.
        infer_rule: ``objectness`` (the tokens that pass, per-image count) or ``topk``
            (``num_queries`` best by objectness).
        local_attn_k / attn_logn_scale / attn_logn_base / min_sample_cells / min_refine_cells /
            anchor_grid_size: see ``DFINETransformer``.
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
        num_queries=300,
        assign_metric="gaussian",
        assign_k=4,
        assign_radius=1,
        min_queries=300,
        obj_threshold=0.5,
        infer_rule="objectness",
        local_attn_k=0,
        attn_logn_scale=False,
        attn_logn_base=None,
        min_sample_cells=0.0,
        min_refine_cells=0.0,
        anchor_grid_size=0.05,
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
            num_queries=num_queries,
            local_attn_k=local_attn_k,
            attn_logn_scale=attn_logn_scale,
            attn_logn_base=attn_logn_base,
            min_sample_cells=min_sample_cells,
            min_refine_cells=min_refine_cells,
            anchor_grid_size=anchor_grid_size,
        )
        assert assign_metric in ("gaussian", "nwd", "giou", "iou"), assign_metric
        assert assign_k >= 1 and assign_radius >= 0
        assert infer_rule in ("objectness", "topk"), infer_rule
        assert min_queries <= num_queries and 0 < obj_threshold < 1
        self.assign_metric = assign_metric
        self.assign_k = assign_k
        self.assign_radius = assign_radius
        self.min_queries = min_queries
        self.obj_threshold = obj_threshold
        self.obj_logit = math.log(obj_threshold / (1 - obj_threshold))  # the threshold on the logit
        self.infer_rule = infer_rule
        # diagnostics of the last training forward, per image: ground truths, forced tokens, how
        # many of them the objectness already lets through, the rule-selected queries and the
        # forced tokens per level
        self.last_assign_stats = None

    # ------------------------------------------------------------------ forced tokens

    def _nearest(self, gt_cxcywh, gt_valid, boxes_of, n, valid, spatial_shapes):
        """
        Each ground truth's ``assign_k`` best tokens, ``[B, M, K]`` indices (best first), over
        ``gt_cxcywh [B, M, 4]`` (``gt_valid [B, M]`` marks padding) and the ``n`` tokens, whose
        predicted boxes ``boxes_of(index)`` gives for ``[B, X]`` token indices (only the
        candidates' are ever computed). The candidates are the tokens of the window of
        ``2 * assign_radius + 1`` cells around the ground truth's centre cell (clamped to the
        grid) on every level, found by index arithmetic; invalid tokens never qualify. A token
        index of ``n`` in the result is a dummy (a padded ground truth's, or a window short of
        ``assign_k`` valid tokens).
        """
        b, m = gt_cxcywh.shape[:2]
        device = gt_cxcywh.device
        r = self.assign_radius
        i32 = torch.int32  # the index arithmetic (N < 2^31); int64 only where gather needs it
        offs = torch.arange(-r, r + 1, dtype=i32, device=device)
        dy, dx = torch.meshgrid(offs, offs, indexing="ij")
        dy, dx = dy.reshape(-1), dx.reshape(-1)  # [P], P = (2r+1)^2

        # the windows of every level at once (the level sizes reach the device without a stream sync)
        hw = torch.tensor(spatial_shapes, dtype=i32).to(device, non_blocking=True)  # [L, 2]
        h, w = hw[:, 0], hw[:, 1]
        start = (h * w).cumsum(0, dtype=i32) - h * w  # each level's first token, [L]
        # the centre cell, clamped to the grid: a centre on the far border lies in the last cell
        col = torch.minimum((gt_cxcywh[..., 0, None] * w).to(i32).clamp_(min=0), w - 1)  # [B, M, L]
        row = torch.minimum((gt_cxcywh[..., 1, None] * h).to(i32).clamp_(min=0), h - 1)
        cc, rr = col[..., None] + dx, row[..., None] + dy  # [B, M, L, P]
        inside = (cc >= 0) & (cc < w[:, None]) & (rr >= 0) & (rr < h[:, None])
        idx = rr.mul_(w[:, None]).add_(cc).add_(start[:, None])
        cand = torch.where(inside, idx, n).flatten(2).long()  # [B, M, C]; n is the dummy token

        # the metric's features of the candidates (the dummy's box is whatever the clamped index
        # gives; it is masked below) and of the ground truths, [B, M, C, 4]
        pred_cxcywh = boxes_of(cand.flatten(1).clamp(max=n - 1)).view(b, m, -1, 4)
        if self.assign_metric == "gaussian":
            feat, gt_feat = pred_cxcywh, gt_cxcywh
        elif self.assign_metric == "nwd":
            # exp(-W2 / C) is monotone in the Wasserstein distance between the boxes' Gaussians, so
            # ranking by the plain distance in (cx, cy, w/2, h/2) space is the same ranking
            feat = torch.cat([pred_cxcywh[..., :2], pred_cxcywh[..., 2:] / 2], dim=-1)
            gt_feat = torch.cat([gt_cxcywh[..., :2], gt_cxcywh[..., 2:] / 2], dim=-1)
        else:
            feat, gt_feat = box_cxcywh_to_xyxy(pred_cxcywh), box_cxcywh_to_xyxy(gt_cxcywh)
        if self.assign_metric == "gaussian":
            match = gaussian_box_similarity(gt_feat[:, :, None, :], feat)
        elif self.assign_metric == "nwd":
            match = -(feat - gt_feat[:, :, None, :]).norm(dim=-1)
        else:
            iou, giou = _pair_iou_giou(gt_feat[:, :, None, :], feat)
            match = iou if self.assign_metric == "iou" else giou
        ok = torch.cat([valid, valid.new_zeros(1)])[cand] & gt_valid[..., None]  # dummy, invalid, padding out
        match = match.masked_fill_(~ok, float("-inf"))
        best = match.topk(min(self.assign_k, match.shape[-1]), dim=-1)
        return torch.where(best.values.isfinite(), cand.gather(-1, best.indices), n)

    @torch.no_grad()
    def _forced(self, targets, boxes_of, n, valid, spatial_shapes):
        """
        The forced tokens ``[B, N]`` of a batch, every real ground truth's ``assign_k`` nearest
        (see ``_nearest``), and the number of real ground truths per image, over the ``n`` tokens
        whose predicted boxes ``boxes_of`` gives. No assignment: a token several ground truths
        want is forced once, and the criterion's matching decides who predicts what.
        """
        b = len(targets)
        device = valid.device
        num_gts = [t["boxes"].shape[0] for t in targets]
        gt_boxes = pad_sequence([t["boxes"] for t in targets], batch_first=True)  # [B, M, 4]
        counts = torch.tensor(num_gts).to(device, non_blocking=True)
        gt_valid = torch.arange(gt_boxes.shape[1], device=device)[None, :] < counts[:, None]
        nearest = self._nearest(gt_boxes, gt_valid, boxes_of, n, valid, spatial_shapes)  # [B, M, K]
        forced = torch.zeros((b, n + 1), dtype=torch.bool, device=device)  # column n: the dummies'
        forced.scatter_(1, nearest.flatten(1), True)
        return forced[:, :n], num_gts

    # ------------------------------------------------------------------ query selection

    def _get_decoder_input(self, memory, spatial_shapes, encoder_out, targets=None):
        """
        Training (with ground truths in the batch): every ground truth's forced tokens plus the
        unforced tokens that pass the objectness threshold, filled up to ``min_queries`` and
        capped at ``num_queries`` by the logit, padded to the largest count in the batch, plus
        the encoder's predictions on the queries for the criterion. Inference: the tokens that
        pass the threshold, clamped to ``[min_queries, num_queries]`` by the logit, or the plain
        top-k by objectness with ``infer_rule='topk'``.
        """
        training = self.training and targets is not None and max(t["boxes"].shape[0] for t in targets) > 0
        if not training and self.infer_rule == "topk":
            return super()._get_decoder_input(memory, spatial_shapes, encoder_out, targets)

        b, n = memory.shape[:2]
        device = memory.device
        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=device)
        valid = valid_mask.reshape(-1)  # [N]
        anchors = anchors.expand(b, -1, -1)  # [B, N, 4], a view

        def take(x, index):
            return x.gather(1, index.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        def select(ranking, count, *extra):
            """
            Every image's ``count`` best tokens by ``ranking``: the query index ``[B, max_count]``
            and its padding mask, plus the counts and every ``extra`` ``[B]`` tensor as lists, read
            from the device in one host sync.
            """
            batch_queries_num, *extra = torch.stack([count, *extra]).tolist()
            max_count = max(batch_queries_num)
            index = ranking.topk(max_count, dim=1).indices
            pad = torch.arange(max_count, device=device)[None, :] >= count[:, None]
            return index, pad, batch_queries_num, *extra

        with torch.no_grad():
            # every token's projected content and objectness (the highest class logit, invalid
            # tokens -inf), without a graph. The invalid tokens are not zeroed first as in D-FINE:
            # none is ever selected, so their outputs go unread (a [B, N, D] copy saved)
            output_memory = self.enc_output(memory)
            scores = self.enc_score_head(output_memory).max(-1).values.masked_fill(~valid, float("-inf"))
            passed = scores > self.obj_logit  # [B, N], the tokens that pass
            num_valid = valid.sum()  # the rule never selects an invalid token
            floor, cap = min(self.min_queries, n), min(self.num_queries, n)

            def boxes_unact_of(index):
                return self.enc_bbox_head(take(output_memory, index)) + take(anchors, index)

            if not training:
                if self.training:
                    self.last_assign_stats = None  # a batch without ground truths
                count = passed.sum(1).clamp(floor, cap).minimum(num_valid)
                index, pad, batch_queries_num = select(scores, count)
                contents = take(output_memory, index)
                return DecoderInput(
                    contents.masked_fill(pad[..., None], 0),
                    boxes_unact_of(index).masked_fill(pad[..., None], 0),
                    [],
                    [],
                    batch_queries_num,
                )

            # the box head on the candidates alone, not on every token (a 960x960 input has 76k)
            forced, num_gts = self._forced(
                targets, lambda index: F.sigmoid(boxes_unact_of(index)), n, valid, spatial_shapes
            )  # [B, N]
            num_forced = forced.sum(1)
            num_selected = (forced & passed).sum(1)  # forced tokens the head lets through
            per_level = torch.stack([f.sum(1) for f in forced.split([h * w for h, w in spatial_shapes], 1)], 1)
            # the forced tokens plus the unforced ones that pass, floored at min_queries and
            # capped at num_queries (a crowd whose forced tokens exceed it keeps them all; they
            # are valid by construction)
            count = (num_forced + (passed & ~forced).sum(1)).clamp(min=floor).minimum(num_forced.clamp(min=cap))
            count = count.minimum(num_valid)
            index, pad, batch_queries_num, num_forced, selected, *levels = select(
                scores.masked_fill(forced, float("inf")), count, num_forced, num_selected, *per_level.unbind(1)
            )
            self.last_assign_stats = [
                {"num_gt": g, "forced": f, "selected": s, "rule": q - f, "levels": lv}
                for g, f, s, q, *lv in zip(num_gts, num_forced, selected, batch_queries_num, *levels)
            ]
        # under autocast, the no-grad calls above cached the heads' half-precision weights without
        # a graph; the calls below would reuse them and reach no parameter
        torch.clear_autocast_cache()

        # the projection and the heads on the queries alone, with a graph: the queries' contents
        # and boxes (detached) and D-FINE's enc_aux_outputs, which the criterion matches (padding
        # is masked there)
        queries = take(memory, index) * valid[index].to(memory.dtype)[..., None]
        output = self.enc_output(queries)
        enc_logits = self.enc_score_head(output)
        boxes_unact = self.enc_bbox_head(output) + take(anchors, index)
        contents = output.detach().masked_fill(pad[..., None], 0)
        boxes_init = boxes_unact.detach().masked_fill(pad[..., None], 0)
        return DecoderInput(contents, boxes_init, [F.sigmoid(boxes_unact)], [enc_logits], batch_queries_num)
