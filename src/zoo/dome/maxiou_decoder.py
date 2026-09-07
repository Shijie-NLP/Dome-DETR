"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

Ground-truth-forced query selection (working name ``MaxIoUTransformer``): a ``DFINETransformer``
whose query selection replaces the top-k by objectness, with no denoising queries. Everything
past the selection is D-FINE's: the encoder's predictions on the queries and the decoder's
outputs are Hungarian-matched by the criterion. The one difference in the losses is on the
encoder side, where the class logits are trained as plain 0/1 objectness (``loss_obj``, no
IoU-aware target, ``enc_losses`` of ``DomeCriterion``), so that their decision boundary,
logit 0, can serve as the selection rule.

- The objectness of a token is its highest class logit; logit 0 is the selection rule, no
  threshold to estimate or store.
- Training (with ground truths in the batch): every ground truth forces one token of its own
  into the queries, the token whose predicted box matches it best (``assign_metric``: ``nwd``,
  ``giou`` or ``iou``) among the ``(2 * assign_radius + 1)``-cell windows around its centre cell
  on every level (index arithmetic, no ``cdist``). Two ground truths wanting the same token are
  not a matter of assignment (the criterion's Hungarian decides who predicts what) but of
  coverage: every ground truth gets a distinct token, so the second one moves on to its next
  candidate (``assign_candidates`` per ground truth; the rare one whose candidates all went to
  others takes the nearest free token). The rest of the queries are the unforced tokens the
  objectness passes, and the count is clamped to ``[min_queries, num_queries]``. Images differ
  in query count (padded to the largest, ``batch_queries_num`` tells the criterion).
- Inference: the tokens the objectness passes, clamped to ``[min_queries, num_queries]`` by its
  logit, so the query count is per image; ``num_queries`` is only a memory guard. As the head
  learns to pass the forced tokens, the training set converges to the inference set plus a
  vanishing forced part. ``infer_rule='topk'`` restores the plain top-k by objectness.
- ``last_assign_stats`` reports, per image, the ground truths, how many forced tokens the head
  already lets through (``selected``, its recall in training), the rule-selected queries and the
  forced tokens per level (``levels``: a drift of tiny objects' tokens towards coarse levels shows
  here).

Host syncs: one per batch (the query counts), plus one boolean when a ground truth needs the
nearest-free-token fallback.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn.utils.rnn import pad_sequence

from ...core import register
from ...misc.box_ops import box_cxcywh_to_xyxy
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
        assign_metric: how a ground truth ranks the tokens' predicted boxes: ``nwd`` (a small box
            that just misses a tiny ground truth still beats a coarse box that merely contains
            it), ``giou`` or ``iou`` (under both, any box containing the ground truth beats any
            box not overlapping it).
        assign_candidates: tokens each ground truth keeps as candidates, its fallbacks when
            another ground truth took its best one.
        assign_radius: cells around the ground truth's centre cell, per level, that are candidates
            (1: a 3x3 window on each level).
        min_queries: the least queries an image gets, in training (forced tokens included) and
            at inference alike.
        infer_rule: ``objectness`` (the head's decision, per-image count) or ``topk``
            (``num_queries`` best by objectness).
        local_attn_k / attn_logn_scale / attn_logn_base: see ``DFINETransformer``.
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
        assign_metric="nwd",
        assign_candidates=8,
        assign_radius=1,
        min_queries=300,
        infer_rule="objectness",
        local_attn_k=0,
        attn_logn_scale=False,
        attn_logn_base=None,
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
        )
        assert assign_metric in ("giou", "iou", "nwd"), assign_metric
        assert assign_candidates >= 1 and assign_radius >= 0
        assert infer_rule in ("objectness", "topk"), infer_rule
        assert min_queries <= num_queries
        self.assign_metric = assign_metric
        self.assign_candidates = assign_candidates
        self.assign_radius = assign_radius
        self.min_queries = min_queries
        self.infer_rule = infer_rule
        # diagnostics of the last training forward, per image: ground truths, how many of their
        # forced tokens the objectness already lets through, the rule-selected queries and the
        # forced tokens per level
        self.last_assign_stats = None

    # ------------------------------------------------------------------ forced tokens

    def _candidates(self, gt_cxcywh, gt_valid, pred_cxcywh, valid, spatial_shapes):
        """
        Each ground truth's ``assign_candidates`` best tokens, ``[B, M, K]`` indices and match
        values (best first), over ``gt_cxcywh [B, M, 4]`` (``gt_valid [B, M]`` marks padding) and
        ``pred_cxcywh [B, N, 4]``. The candidates are the tokens of the window of
        ``2 * assign_radius + 1`` cells around the ground truth's centre cell (clamped to the grid)
        on every level, found by index arithmetic; invalid tokens never qualify. A token index of
        ``N`` in the result is a dummy (its value is ``-inf``).
        """
        b, m = gt_cxcywh.shape[:2]
        n = pred_cxcywh.shape[1]
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

        # the metric's features of every token and ground truth, the candidates' gathered through
        # a zero row for the dummy token, [B, M, C, 4]
        if self.assign_metric == "nwd":
            # exp(-W2 / C) is monotone in the Wasserstein distance between the boxes' Gaussians, so
            # ranking by the plain distance in (cx, cy, w/2, h/2) space is the same ranking
            feat = torch.cat([pred_cxcywh[..., :2], pred_cxcywh[..., 2:] / 2], dim=-1)
            gt_feat = torch.cat([gt_cxcywh[..., :2], gt_cxcywh[..., 2:] / 2], dim=-1)
        else:
            feat, gt_feat = box_cxcywh_to_xyxy(pred_cxcywh), box_cxcywh_to_xyxy(gt_cxcywh)
        feat = torch.cat([feat, feat.new_zeros((b, 1, 4))], dim=1)
        feat = feat.gather(1, cand.flatten(1).unsqueeze(-1).expand(-1, -1, 4)).view(b, m, -1, 4)
        if self.assign_metric == "nwd":
            match = -(feat - gt_feat[:, :, None, :]).norm(dim=-1)
        else:
            iou, giou = _pair_iou_giou(gt_feat[:, :, None, :], feat)
            match = iou if self.assign_metric == "iou" else giou
        ok = torch.cat([valid, valid.new_zeros(1)])[cand] & gt_valid[..., None]  # dummy, invalid, padding out
        match = match.masked_fill_(~ok, float("-inf"))
        best = match.topk(min(self.assign_candidates, match.shape[-1]), dim=-1)
        return torch.where(best.values.isfinite(), cand.gather(-1, best.indices), n), best.values

    @torch.no_grad()
    def _forced(self, targets, pred_cxcywh, valid, spatial_shapes):
        """
        The tokens the ground truths force into the queries, a ``[B, N]`` mask with one distinct
        token per real ground truth, and the number of real ground truths per image. Round ``r``
        gives every ground truth still without a token its ``r``-th candidate unless another has
        it (the lowest-index ground truth wins a tie); the rare ground truth whose candidates all
        went to others takes the nearest free token, settled on the host.
        """
        b, n = pred_cxcywh.shape[:2]
        device = pred_cxcywh.device
        num_gts = [t["boxes"].shape[0] for t in targets]
        gt_boxes = pad_sequence([t["boxes"] for t in targets], batch_first=True)  # [B, M, 4]
        m = gt_boxes.shape[1]
        counts = torch.tensor(num_gts).to(device, non_blocking=True)
        gt_valid = torch.arange(m, device=device)[None, :] < counts[:, None]
        cand, _ = self._candidates(gt_boxes, gt_valid, pred_cxcywh, valid, spatial_shapes)  # [B, M, K]

        forced = torch.zeros((b, n + 1), dtype=torch.bool, device=device)  # column n: the dummy's bin
        owner = torch.empty((b, n + 1), dtype=torch.long, device=device)
        gt = torch.arange(m, device=device)[None, :].expand(b, -1)
        left = gt_valid.clone()
        for r in range(cand.shape[-1]):
            tok = cand[..., r]  # [B, M]
            want = left & (tok < n) & ~forced.gather(1, tok)
            owner.fill_(m).scatter_reduce_(1, torch.where(want, tok, n), gt, "amin")
            win = want & (owner.gather(1, tok) == gt)
            forced.scatter_(1, torch.where(win, tok, n), True)
            left &= ~win
        forced = forced[:, :n]

        if left.any():  # one boolean; only when a ground truth's candidates all went to others
            cost = torch.where(valid[None, :] & ~forced, 0.0, float("inf")).to(pred_cxcywh.dtype)
            for i, j in left.nonzero().tolist():
                dist = torch.cdist(gt_boxes[i, j : j + 1, :2], pred_cxcywh[i, :, :2]).squeeze(0) + cost[i]
                t = dist.argmin()
                forced[i, t], cost[i, t] = True, float("inf")
        return forced, num_gts

    # ------------------------------------------------------------------ query selection

    def _encoder_tokens(self, memory, spatial_shapes):
        """
        Every token's projected content, class logits ``[B, N, C]`` (with a graph), its objectness
        (the highest class logit, detached, invalid tokens -inf), anchor and validity.
        """
        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        valid = valid_mask.reshape(-1)  # [N]
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)
        memory = valid_mask.to(memory.dtype) * memory
        output_memory: torch.Tensor = self.enc_output(memory)
        enc_outputs_logits: torch.Tensor = self.enc_score_head(output_memory)  # [B, N, C]
        scores = enc_outputs_logits.detach().max(-1).values.masked_fill(~valid[None, :], float("-inf"))
        return output_memory, enc_outputs_logits, scores, anchors, valid

    def _get_decoder_input(self, memory, spatial_shapes, encoder_out, targets=None):
        """
        Training (with ground truths in the batch): every ground truth's forced token plus the
        unforced tokens the objectness passes, by its logit, padded to the largest count in the
        batch, plus the encoder's predictions on the queries for the criterion. Inference: the
        tokens the objectness passes, clamped to ``[min_queries, num_queries]`` by its logit, or
        the plain top-k by objectness with ``infer_rule='topk'``.
        """
        training = self.training and targets is not None and max(t["boxes"].shape[0] for t in targets) > 0
        if not training and self.infer_rule == "topk":
            return super()._get_decoder_input(memory, spatial_shapes, encoder_out, targets)

        output_memory, enc_outputs_logits, scores, anchors, valid = self._encoder_tokens(memory, spatial_shapes)
        b, n = scores.shape
        device = memory.device

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
            passed = scores > 0  # [B, N], the head's decision
            num_valid = valid.sum()  # the rule never selects an invalid token
            floor, cap = min(self.min_queries, n), min(self.num_queries, n)

            if not training:
                if self.training:
                    self.last_assign_stats = None  # a batch without ground truths
                count = passed.sum(1).clamp(floor, cap).minimum(num_valid)
                index, pad, batch_queries_num = select(scores, count)
                contents = take(output_memory, index)
                boxes_unact = self.enc_bbox_head(contents) + take(anchors, index)  # the head on the queries alone
                return DecoderInput(
                    contents.masked_fill(pad[..., None], 0),
                    boxes_unact.masked_fill(pad[..., None], 0),
                    [],
                    [],
                    batch_queries_num,
                )

            # every token's box: the pick needs them. In fp32 with autocast off: under autocast this
            # no-grad call would cache the head's half-precision weights without a graph, and the
            # graph-bearing call on the queries below would reuse them and reach no parameter
            with torch.autocast(device.type, enabled=False):
                boxes_unact = self.enc_bbox_head(output_memory.float()) + anchors
            forced, num_gts = self._forced(targets, F.sigmoid(boxes_unact), valid, spatial_shapes)  # [B, N]
            num_gt = forced.sum(1)
            num_selected = (forced & passed).sum(1)  # forced tokens the head lets through
            per_level = torch.stack([f.sum(1) for f in forced.split([h * w for h, w in spatial_shapes], 1)], 1)
            # the forced tokens plus the unforced ones the head passes, floored and capped (a
            # crowd beyond num_queries keeps its forced tokens; they are valid by construction)
            count = (num_gt + (passed & ~forced).sum(1)).clamp(min=floor).minimum(num_gt.clamp(min=cap))
            count = count.minimum(num_valid)
            index, pad, batch_queries_num, selected, *levels = select(
                scores.masked_fill(forced, float("inf")), count, num_selected, *per_level.unbind(1)
            )
            contents = take(output_memory, index).masked_fill(pad[..., None], 0)
            boxes_init = take(boxes_unact, index).masked_fill(pad[..., None], 0)
            self.last_assign_stats = [
                {"num_gt": g, "selected": s, "rule": q - g, "levels": lv}
                for g, s, q, *lv in zip(num_gts, selected, batch_queries_num, *levels)
            ]

        # the encoder's predictions on the queries, with a graph: D-FINE's enc_aux_outputs, which
        # the criterion Hungarian-matches (padding is masked there)
        enc_logits = take(enc_outputs_logits, index)
        enc_boxes = F.sigmoid(self.enc_bbox_head(take(output_memory, index)) + take(anchors, index))
        return DecoderInput(contents, boxes_init, [enc_boxes], [enc_logits], batch_queries_num)
