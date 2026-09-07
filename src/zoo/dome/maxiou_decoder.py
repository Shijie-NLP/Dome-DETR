"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

Ground-truth-claimed query initialization (working name ``MaxIoUTransformer``): a
``DFINETransformer`` whose training splits into three parts, with no denoising queries. The
encoder's class scores are trained as plain 0/1 targets and their maximum, an objectness, picks
the queries in training and inference alike; the decoder alone decides localization quality.

- Encoder, dense: every token predicts a box; each ground truth claims the token whose box
  matches it best (``giou`` by default, ``iou`` or ``nwd``), one token per ground truth. The
  encoder's score head keeps its class logits but is trained with a class-balanced BCE on 0/1
  targets, 1 on the claimed token's ground-truth class and 0 on every other (token, class) entry
  (``loss_obj``, no IoU-aware target); the claimed tokens also get the box losses (``enc_dense``).
  No Hungarian on the encoder side. The objectness of a token is its highest class logit, and
  the decision boundary, logit 0, is the selection rule: no threshold to estimate or store.
  ``last_assign_stats`` reports, per image, the claimed tokens the head already lets through
  (``selected``, i.e. its recall in training), how many claims came from the fallback and how
  many claimed tokens lie on each level (``levels``: a drift of tiny objects' claims towards
  coarse levels shows here).
- Decoder queries, training: the claimed tokens (forced) plus every other token the objectness
  selects, floored at ``min_negatives`` of the latter and capped so an image has at most
  ``max(num_queries, #gt + min_negatives)`` queries. Every query is a real token with its own
  content and predicted box; the decoder's Hungarian matching labels them. Images differ in query
  count (padded to the largest, ``batch_queries_num`` tells the criterion).
- Inference: the tokens the objectness selects, clamped to ``[min_queries, num_queries]`` by its
  logit, so the query count is per image; ``num_queries`` is only a memory guard. As the head
  learns to pass the claimed tokens, the training set converges to the inference set plus a
  vanishing forced part. ``infer_rule='topk'`` restores the plain top-k by objectness.

The claim is one-to-one and runs batched over the images with the ground truths padded to the
largest count. Candidates come from the token grid, not from a distance matrix: on every level the
tokens of the ``(2 * assign_radius + 1)``-cell window around the ground truth's centre cell (index
arithmetic, no ``cdist``), scored exactly with ``assign_metric`` (for ``nwd`` the ranking by the
normalized Gaussian Wasserstein distance is the ranking by the plain distance in
``(cx, cy, w/2, h/2)`` space). Each ground truth keeps its ``assign_candidates`` best tokens; when
two ground truths want the same token the better-matching one keeps it and the other moves to its
next candidate, in vectorized rounds until nobody loses (a boolean host sync per round; a few
rounds in practice). A ground truth whose candidates all went to others then
takes the nearest-centre unclaimed token, so every ground truth ends up with a query of its own
(``_fallback``, on the leftovers alone, in rounds). One more host sync per batch (the query
counts) plus one count per fallback round.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn.utils.rnn import pad_sequence

from ...core import register
from ...misc.box_ops import box_cxcywh_to_xyxy
from ...nn.functional import inverse_sigmoid
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
        assign_metric: ``giou`` (default: unlike IoU it still ranks tokens whose box does not
            overlap a tiny ground truth), ``iou`` or ``nwd``.
        assign_candidates: tokens each ground truth keeps as candidates.
        assign_radius: cells around the ground truth's centre cell, per level, that are candidates
            (1: a 3x3 window on each level).
        assign_chunk: ground truths per chunk of the fallback's distance matrix (memory).
        min_negatives: the least rule-selected (unforced) queries an image gets in training.
        min_queries: the least queries an image gets at inference.
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
        assign_metric="giou",
        assign_candidates=8,
        assign_radius=1,
        assign_chunk=256,
        min_negatives=100,
        min_queries=100,
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
        self.assign_chunk = assign_chunk
        self.min_negatives = min_negatives
        self.min_queries = min_queries
        self.infer_rule = infer_rule
        # diagnostics of the last training forward, per image: ground truths, how many of their
        # claimed tokens the objectness already lets through, the rule-selected queries, the claims
        # that needed the fallback and the claimed tokens per level
        self.last_assign_stats = None

    # ------------------------------------------------------------------ ground-truth claims

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

        # the metric's features of the candidate tokens and the ground truths, [B, M, C, 4] (a
        # dummy reads token N - 1 and is masked below)
        token = cand.clamp(max=n - 1)
        if self.assign_metric == "nwd":
            # exp(-W2 / C) is monotone in the Wasserstein distance between the boxes' Gaussians, so
            # ranking by the plain distance in (cx, cy, w/2, h/2) space is the same ranking
            feat = torch.cat([pred_cxcywh[..., :2], pred_cxcywh[..., 2:] / 2], dim=-1)
            gt_feat = torch.cat([gt_cxcywh[..., :2], gt_cxcywh[..., 2:] / 2], dim=-1)
        else:
            feat, gt_feat = box_cxcywh_to_xyxy(pred_cxcywh), box_cxcywh_to_xyxy(gt_cxcywh)
        feat = feat.gather(1, token.flatten(1).unsqueeze(-1).expand(-1, -1, 4)).view(b, m, -1, 4)
        if self.assign_metric == "nwd":
            match = -(feat - gt_feat[:, :, None, :]).norm(dim=-1)
        else:
            iou, giou = _pair_iou_giou(gt_feat[:, :, None, :], feat)
            match = iou if self.assign_metric == "iou" else giou
        ok = inside.flatten(2) & valid[token] & gt_valid[..., None]  # dummy, invalid, padding out
        match = match.masked_fill_(~ok, float("-inf"))
        best = match.topk(min(self.assign_candidates, match.shape[-1]), dim=-1)
        return torch.where(best.values.isfinite(), cand.gather(-1, best.indices), n), best.values

    def _resolve(self, cand_idx, cand_val, gt_valid, n_tokens):
        """
        One-to-one claims from the candidate lists ``[B, M, K]`` (``n_tokens`` marks a dummy): a
        ground truth that loses its current candidate to a better-matching ground truth moves to
        its next one, until nobody loses. Returns the claimed token of every ground truth,
        ``[B, M]``, ``-1`` where the candidates ran out or the ground truth is padding.
        """
        b, m, k = cand_idx.shape
        device = cand_idx.device
        # tokens numbered across the batch (the dummy included); the bid tables are one row per
        # token, a few MiB, so nothing to compact
        u = b * (n_tokens + 1)
        slots = cand_idx + (torch.arange(b, device=device) * (n_tokens + 1))[:, None, None]
        gt = torch.arange(b * m, device=device).view(b, m)
        best = cand_val.new_empty((u,))
        owner = torch.empty((u + 1,), dtype=torch.long, device=device)  # row u: the non-holders' bin
        ptr = torch.zeros((b, m), dtype=torch.long, device=device)
        active = gt_valid.clone()
        # A round: every active ground truth bids on its current candidate; the best bid holds the
        # token and the others move on. Winners can lose later to a better bid that moves in, so
        # the rounds run until nobody loses. Every loss advances a pointer, so k * m rounds bound it.
        for _ in range(k * m + 1):
            slot = slots.gather(-1, ptr[..., None]).squeeze(-1)  # [B, M]
            val = cand_val.gather(-1, ptr[..., None]).squeeze(-1)
            active &= val.isfinite()  # candidates are best first: past a dummy, only dummies remain
            # the best bid on each slot, then the lowest-index ground truth holding it
            best.fill_(float("-inf")).scatter_reduce_(0, slot.flatten(), val.flatten(), "amax")
            holds = active & (val >= best[slot])
            owner.fill_(b * m).scatter_reduce_(0, torch.where(holds, slot, u).flatten(), gt.flatten(), "amin")
            win = holds & (owner[slot] == gt)
            lose = active & ~win
            if not lose.any():  # one boolean per round
                break
            ptr = ptr + lose.long()
            active &= ptr < k
            ptr = ptr.clamp(max=k - 1)
        # the winners hold distinct tokens by construction
        tok = cand_idx.gather(-1, ptr[..., None]).squeeze(-1)
        return torch.where(win, tok, torch.full_like(tok, -1))

    @torch.no_grad()
    def _claim(self, targets, pred_cxcywh, valid, spatial_shapes):
        """
        The tokens the ground truths claim: ``[B, M]`` token indices in ground-truth order (``-1``
        only for padded ground truths), the padded ground-truth boxes ``[B, M, 4]`` and their
        validity ``[B, M]``, the number of real ground truths per image and how many of them
        needed the fallback, ``[B]``.
        """
        n = pred_cxcywh.shape[1]
        device = pred_cxcywh.device
        num_gts = [t["boxes"].shape[0] for t in targets]
        gt_boxes = pad_sequence([t["boxes"] for t in targets], batch_first=True).to(pred_cxcywh.dtype)
        gt_valid = (
            torch.arange(gt_boxes.shape[1], device=device)[None, :] < torch.tensor(num_gts, device=device)[:, None]
        )
        cand_idx, cand_val = self._candidates(gt_boxes, gt_valid, pred_cxcywh, valid, spatial_shapes)
        assigned = self._resolve(cand_idx, cand_val, gt_valid, n)
        fallen = ((assigned < 0) & gt_valid).sum(1)  # ground truths whose candidates all went to others
        assigned = self._fallback(assigned, gt_boxes, gt_valid, pred_cxcywh, valid)
        return assigned, gt_boxes, gt_valid, num_gts, fallen

    def _fallback(self, assigned, gt_cxcywh, gt_valid, pred_cxcywh, valid):
        """
        Every real ground truth gets a token: one whose candidates all went to better-matching
        ground truths takes the nearest-centre token nobody claimed. Works on the leftovers alone:
        each keeps its ``assign_candidates`` nearest unclaimed tokens and the claims are resolved
        as before, in rounds until every leftover holds a token (the nearest wins a contested
        token). Only runs when needed.
        """
        b, n = pred_cxcywh.shape[:2]
        device = assigned.device
        k = min(self.assign_candidates, n)
        taken = torch.zeros((b, n + 1), dtype=torch.bool, device=device)
        rounds = None
        while True:
            left = (assigned < 0) & gt_valid
            num_left = left.sum(1)
            max_left = int(num_left.max())  # one host sync per round
            if max_left == 0:
                return assigned
            if rounds is None:
                rounds = max_left  # every round settles the best bidder of each image with leftovers
            assert rounds > 0, "an image has more ground truths than valid tokens"
            rounds -= 1
            taken.scatter_(1, torch.where(assigned >= 0, assigned, n), True)
            blocked = torch.where(valid[None, :] & ~taken[:, :n], 0.0, float("inf")).to(pred_cxcywh.dtype)
            # the leftovers compacted to [B, max_left], in ground-truth order, then their nearest
            # unclaimed tokens (cdist's matmul path is exact enough: its error is ~1e-7 in the
            # squared distance, the cells are 1e-2 wide)
            order = torch.sort((~left).long(), dim=1, stable=True).indices[:, :max_left]
            left_valid = torch.arange(max_left, device=device)[None, :] < num_left[:, None]
            centres = gt_cxcywh[..., :2].gather(1, order[..., None].expand(-1, -1, 2))
            idx, val = [], []
            for start in range(0, max_left, self.assign_chunk):
                sl = slice(start, start + self.assign_chunk)
                dist = torch.cdist(centres[:, sl], pred_cxcywh[..., :2])
                near = dist.add_(blocked[:, None, :]).topk(k, dim=-1, largest=False)
                idx.append(near.indices)
                val.append(-near.values)  # blocked tokens: -inf, never held
            val = torch.cat(val, 1).masked_fill(~left_valid[..., None], float("-inf"))
            claimed = self._resolve(torch.cat(idx, 1), val, left_valid, n)  # [B, max_left]
            assigned = assigned.scatter(1, order, torch.where(left_valid, claimed, assigned.gather(1, order)))

    # ------------------------------------------------------------------ query initialization

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
        Training (with ground truths in the batch): every ground truth's claimed token, then the
        other tokens the objectness passes, by its logit, padded to the largest count in the
        batch, plus the encoder's dense outputs for the criterion (``extra``). Inference: the
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

        def layout(forced, num_forced, rule_count, *extra):
            """
            The query index ``[B, max_total]`` and padding mask: each image's forced tokens
            (``forced[i, :num_forced[i]]``, a prefix of real token indices), then its ``rule_count``
            best tokens by objectness among the rest; the real counts and ``rule_count``, and every
            ``extra`` ``[B]`` tensor, as lists read from the device in the same, single host sync.
            """
            m = forced.shape[1]
            taken = torch.zeros((b, n + 1), dtype=torch.bool, device=device)
            taken.scatter_(1, torch.where(forced >= 0, forced, n), True)
            rest = scores.masked_fill(taken[:, :n], float("-inf"))
            total = num_forced + rule_count
            batch_queries_num, rule_list, *extra = torch.stack([total, rule_count, *extra]).tolist()
            max_total, k = max(batch_queries_num), min(max(rule_list), n)
            ranked = rest.topk(k, dim=1).indices
            pos = torch.arange(max_total, device=device)[None, :]
            pad = pos >= total[:, None]
            index = ranked.gather(1, (pos - num_forced[:, None]).clamp(0, k - 1).expand(b, -1))
            if m:  # the forced prefix; padded rows of ``forced`` are never read
                from_forced = forced.gather(1, pos.clamp(max=m - 1).expand(b, -1))
                index = torch.where(pos < num_forced[:, None], from_forced, index)
            return index, pad, batch_queries_num, rule_list, *extra

        with torch.no_grad():
            all_boxes = F.sigmoid(self.enc_bbox_head(output_memory) + anchors)  # no graph over all tokens
            passed = scores > 0  # [B, N], the head's decision

            if not training:
                count = passed.sum(1).clamp(min(self.min_queries, n), min(self.num_queries, n))
                index, pad, batch_queries_num, _ = layout(
                    scores.new_empty((b, 0), dtype=torch.long), torch.zeros_like(count), count
                )
                contents = take(output_memory, index).masked_fill(pad[..., None], 0)
                boxes_unact = inverse_sigmoid(take(all_boxes, index)).masked_fill(pad[..., None], 0)
                return DecoderInput(contents, boxes_unact, [], [], batch_queries_num)

            assigned, gt_boxes, gt_valid, num_gts, fallen = self._claim(targets, all_boxes, valid, spatial_shapes)
            num_gt = gt_valid.sum(1)
            claimed = assigned.clamp(min=0)  # padding rows point at token 0 and are masked below
            claimed_passed = (take(scores.unsqueeze(-1), claimed).squeeze(-1) > 0) & gt_valid
            # the claimed tokens per level, [B, L]
            ends = torch.tensor([h * w for h, w in spatial_shapes]).cumsum(0).to(device, non_blocking=True)
            level = F.one_hot(torch.bucketize(claimed, ends, right=True), len(spatial_shapes))
            per_level = level.masked_fill(~gt_valid[..., None], 0).sum(1)

            # rule-selected queries: the unclaimed tokens the objectness passes, floored and capped
            passed_free = passed.sum(1) - claimed_passed.sum(1)
            cap = (self.num_queries - num_gt).clamp(min=self.min_negatives)
            rule_count = passed_free.clamp(min=self.min_negatives).minimum(cap).clamp(max=n - num_gt)
            index, pad, batch_queries_num, rule_list, selected, fell, *levels = layout(
                assigned, num_gt, rule_count, claimed_passed.sum(1), fallen, *per_level.unbind(1)
            )

            contents = take(output_memory, index).masked_fill(pad[..., None], 0)
            boxes_unact = inverse_sigmoid(take(all_boxes, index)).masked_fill(pad[..., None], 0)

            self.last_assign_stats = [
                {"num_gt": g, "selected": s, "rule": c, "fallback": f, "levels": lv}
                for g, s, c, f, *lv in zip(num_gts, selected, rule_list, fell, *levels)
            ]

        # the encoder's dense outputs: every token's class logits, and the claimed tokens' boxes
        # with a graph; the criterion trains both against the claim
        claimed_bbox = F.sigmoid(self.enc_bbox_head(take(output_memory, claimed)) + take(anchors, claimed))
        dense_boxes = all_boxes.scatter(
            1,
            claimed.unsqueeze(-1).expand(-1, -1, 4),
            torch.where(gt_valid[..., None], claimed_bbox, take(all_boxes, claimed)),
        )
        enc_dense = {
            "pred_logits": enc_outputs_logits,
            "pred_boxes": dense_boxes,
            "valid": valid,
            "indices": [(assigned[i, : num_gts[i]], torch.arange(num_gts[i], device=device)) for i in range(b)],
        }
        return DecoderInput(contents, boxes_unact, [], [], batch_queries_num, extra={"enc_dense": enc_dense})
