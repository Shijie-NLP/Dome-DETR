"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from scipy.optimize import linear_sum_assignment
from torch.nn.utils.rnn import pad_sequence

from ...core import register
from ...misc.box_ops import box_cxcywh_to_xyxy, elementwise_generalized_box_iou, gaussian_box_similarity

__all__ = ["HungarianMatcher", "padded_targets", "topk_matching"]

# the assignments of a batch run in parallel: scipy releases the GIL in linear_sum_assignment
_ASSIGN_POOL = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1))


@torch.no_grad()
def padded_targets(targets, num_queries, batch_queries_num=None):
    """
    A batch's targets as padded tensors: labels ``[B, M]`` and boxes ``[B, M, 4]`` (zeros past
    each image's count), the ground truths' validity ``[B, M]`` and the queries' ``[B, Q]``
    (every query, or the first ``batch_queries_num[i]`` of image ``i``). ``M`` is 0 for a batch
    without ground truths.
    """
    device = targets[0]["boxes"].device
    counts = torch.tensor([t["boxes"].shape[0] for t in targets])
    m = int(counts.max())
    if m:
        labels = pad_sequence([t["labels"] for t in targets], batch_first=True)
        boxes = pad_sequence([t["boxes"] for t in targets], batch_first=True)
    else:
        labels = torch.zeros((len(targets), 0), dtype=torch.long, device=device)
        boxes = torch.zeros((len(targets), 0, 4), dtype=targets[0]["boxes"].dtype, device=device)
    gt_valid = torch.arange(m, device=device)[None, :] < counts.to(device, non_blocking=True)[:, None]
    queries = torch.arange(num_queries, device=device)[None, :]
    if batch_queries_num is None:
        q_valid = torch.ones((len(targets), num_queries), dtype=torch.bool, device=device)
    else:
        q_valid = queries < torch.tensor(batch_queries_num).to(device, non_blocking=True)[:, None]
    return labels, boxes, gt_valid, q_valid


def _split_pairs(query_idx, target_idx, batch_idx, num_images):
    """Per-image ``(pred_idx, target_idx)`` pairs from flat pairs sorted by image, one host sync."""
    per_image = torch.bincount(batch_idx, minlength=num_images).tolist()
    return list(zip(query_idx.split(per_image), target_idx.split(per_image)))


@torch.no_grad()
def topk_matching(outputs, targets, k, batch_queries_num=None):
    """
    One-to-many assignment for a prediction set whose predictions are candidates rather than
    detections (the encoder's queries): every ground truth takes its ``k`` most similar real
    queries by ``box_ops.gaussian_box_similarity`` of the boxes, and a query wanted by several
    ground truths goes to the one it is most similar to, so a query has at most one target and a
    ground truth at most ``k`` queries. Returns per-image ``(pred_idx, target_idx)`` pairs on the
    predictions' device, as ``HungarianMatcher.forward`` does. The batch is assigned in one go:
    padded queries and ground truths take part with similarity -inf and are dropped at the end.
    """
    boxes = outputs["pred_boxes"]  # [B, Q, 4]
    b, q = boxes.shape[:2]
    empty = torch.zeros(0, dtype=torch.long, device=boxes.device)
    _, gt, gt_valid, q_valid = padded_targets(targets, q, batch_queries_num)
    if gt.shape[1] == 0 or q == 0:
        return [(empty, empty) for _ in range(b)]
    sim = gaussian_box_similarity(boxes[:, :, None, :], gt[:, None, :, :])  # [B, Q, M]
    sim = sim.masked_fill(~(q_valid[:, :, None] & gt_valid[:, None, :]), float("-inf"))
    wanted = torch.zeros_like(sim, dtype=torch.bool).scatter_(1, sim.topk(min(k, q), dim=1).indices, True)
    wanted &= gt_valid[:, None, :]  # a padded ground truth wants nothing
    best = sim.masked_fill(~wanted, -1.0).max(dim=2)  # each query's best ground truth among those wanting it
    pairs = (best.values >= 0).nonzero()  # [K, 2] (image, query), sorted by image
    batch_idx, src = pairs.unbind(1)
    return _split_pairs(src, best.indices[batch_idx, src], batch_idx, b)


@register()
class HungarianMatcher(nn.Module):
    """
    One-to-one assignment of predictions to ground-truth boxes, per image, by the Hungarian
    algorithm on a cost of ``cost_class * class + cost_bbox * L1 + cost_giou * (-GIoU) +
    cost_gaussian * (1 - Gaussian similarity)`` (``weight_dict``; a missing or zero weight
    skips the term). The class cost is the focal-style cost of the target class when
    ``use_focal_loss`` is set (the config's top-level flag), else ``-softmax probability``. The
    Gaussian term (``box_ops.gaussian_box_similarity``) still tells apart candidates that do not
    overlap a tiny ground truth, where GIoU saturates and the normalized L1 is negligible.
    Predictions left unmatched are background.
    """

    __share__ = ["use_focal_loss"]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = weight_dict.get("cost_class", 0)
        self.cost_bbox = weight_dict.get("cost_bbox", 0)
        self.cost_giou = weight_dict.get("cost_giou", 0)
        self.cost_gaussian = weight_dict.get("cost_gaussian", 0)
        assert any((self.cost_class, self.cost_bbox, self.cost_giou, self.cost_gaussian)), "all costs cant be 0"

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

    def _cost(self, logits, boxes, tgt_ids, tgt_bbox):
        """
        The cost matrices ``[B, Q, M]`` of a batch's queries (``logits [B, Q, C]``, ``boxes
        [B, Q, 4]``) against its padded ground truths (``tgt_ids [B, M]``, ``tgt_bbox [B, M, 4]``);
        the entries of padded queries and ground truths are meaningless and left to the caller.
        """
        prob = F.sigmoid(logits) if self.use_focal_loss else logits.softmax(-1)
        out_prob = prob.gather(-1, tgt_ids[:, None, :].expand(-1, logits.shape[1], -1))  # the target class's
        if self.use_focal_loss:
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob

        cost = self.cost_class * cost_class + self.cost_bbox * torch.cdist(boxes, tgt_bbox, p=1)
        if self.cost_giou:
            giou = elementwise_generalized_box_iou(
                box_cxcywh_to_xyxy(boxes)[:, :, None, :], box_cxcywh_to_xyxy(tgt_bbox)[:, None, :, :]
            )
            cost = cost - self.cost_giou * giou
        if self.cost_gaussian:
            cost = cost + self.cost_gaussian * (
                1 - gaussian_box_similarity(boxes[:, :, None, :], tgt_bbox[:, None, :, :])
            )
        return cost

    @torch.no_grad()
    def forward(self, outputs: dict[str, torch.Tensor], targets, batch_queries_num=None):
        """
        Args:
            outputs: ``pred_logits`` ``[B, Q, C]`` and ``pred_boxes`` ``[B, Q, 4]`` (normalized cxcywh).
            targets: one dict per image with ``labels`` ``[N_i]`` and ``boxes`` ``[N_i, 4]``.
            batch_queries_num: the real query count of every image; the queries past it are
                padding and never matched.

        Returns:
            ``{"indices": [(pred_idx, target_idx), ...]}``, one pair of int64 index tensors per
            image on the predictions' device, each of length ``min(Q_i, N_i)``.
        """
        return {"indices": self.match_sets([outputs], targets, batch_queries_num)[0]}

    @torch.no_grad()
    def match_sets(self, outputs_list, targets, batch_queries_num=None):
        """
        The matching of several prediction sets (each as in ``forward``, all with the same number
        of queries) against the same targets in one go: the costs are computed on the device a
        set at a time over the whole batch (the padded queries and ground truths dropped
        afterwards), reach the host in one copy, the assignments run in parallel threads and the
        indices go back in one copy. Returns one list of per-image index pairs per set.
        """
        device = outputs_list[0]["pred_logits"].device
        b = len(targets)
        q = outputs_list[0]["pred_logits"].shape[1]
        tgt_ids, tgt_bbox, gt_valid, q_valid = padded_targets(targets, q, batch_queries_num)
        real = q_valid[:, :, None] & gt_valid[:, None, :]  # [B, Q, M]
        costs = []
        for outputs in outputs_list:
            assert outputs["pred_logits"].shape[1] == q, "every set has the queries of the first"
            cost = self._cost(outputs["pred_logits"], outputs["pred_boxes"], tgt_ids, tgt_bbox)
            costs.append(cost[real])  # image by image, each row-major: the sub-matrices back to back
        num_q, num_gt = q_valid.sum(1).tolist(), gt_valid.sum(1).tolist()
        shapes = [(nq, ng) for _ in outputs_list for nq, ng in zip(num_q, num_gt)]
        flat = torch.nan_to_num(torch.cat(costs), nan=1.0).cpu()  # the one device-to-host copy
        mats = [m.view(shape).numpy() for m, shape in zip(flat.split([nq * ng for nq, ng in shapes]), shapes)]
        pairs = list(_ASSIGN_POOL.map(linear_sum_assignment, mats))

        lengths = [len(i) for i, _ in pairs]
        packed = torch.from_numpy(np.concatenate([np.stack([i, j]) for i, j in pairs], axis=1)).to(device)
        rows, cols = packed[0].split(lengths), packed[1].split(lengths)
        indices = list(zip(rows, cols))
        return [indices[k * b : (k + 1) * b] for k in range(len(outputs_list))]
