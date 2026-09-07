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

from ...core import register
from ...misc.box_ops import box_cxcywh_to_xyxy, gaussian_box_similarity, generalized_box_iou

__all__ = ["HungarianMatcher", "topk_matching"]

# the assignments of a batch run in parallel: scipy releases the GIL in linear_sum_assignment
_ASSIGN_POOL = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1))


@torch.no_grad()
def topk_matching(outputs, targets, k, batch_queries_num=None):
    """
    One-to-many assignment for a prediction set whose predictions are candidates rather than
    detections (the encoder's queries): every ground truth takes its ``k`` most similar real
    queries by ``box_ops.gaussian_box_similarity`` of the boxes, and a query wanted by several
    ground truths goes to the one it is most similar to, so a query has at most one target and a
    ground truth at most ``k`` queries. Returns per-image ``(pred_idx, target_idx)`` pairs on the
    predictions' device, as ``HungarianMatcher.forward`` does.
    """
    boxes = outputs["pred_boxes"]
    empty = torch.zeros(0, dtype=torch.long, device=boxes.device)
    indices = []
    for i, t in enumerate(targets):
        q = boxes.shape[1] if batch_queries_num is None else batch_queries_num[i]
        gt = t["boxes"]
        if gt.shape[0] == 0 or q == 0:
            indices.append((empty, empty))
            continue
        sim = gaussian_box_similarity(boxes[i, :q, None, :], gt[None, :, :])  # [Q_i, N_i]
        wanted = torch.zeros_like(sim, dtype=torch.bool).scatter_(0, sim.topk(min(k, q), dim=0).indices, True)
        best = sim.masked_fill(~wanted, -1.0).max(dim=1)  # each query's best ground truth among those wanting it
        src = (best.values >= 0).nonzero().squeeze(1)
        indices.append((src, best.indices[src]))
    return indices


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
        """The cost matrix ``[Q, N]`` of one image's real queries against its ground truths."""
        if self.use_focal_loss:
            out_prob = F.sigmoid(logits)[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -logits.softmax(-1)[:, tgt_ids]

        cost = self.cost_class * cost_class + self.cost_bbox * torch.cdist(boxes, tgt_bbox, p=1)
        if self.cost_giou:
            cost = cost - self.cost_giou * generalized_box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(tgt_bbox))
        if self.cost_gaussian:
            cost = cost + self.cost_gaussian * (1 - gaussian_box_similarity(boxes[:, None, :], tgt_bbox[None, :, :]))
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
        The matching of several prediction sets (each as in ``forward``) against the same targets
        in one go: the cost of every (set, image) is computed on the device over the image's real
        queries alone, the costs reach the host in one copy, the assignments run in parallel
        threads and the indices go back in one copy. Returns one list of per-image index pairs
        per set.
        """
        device = outputs_list[0]["pred_logits"].device
        costs, shapes = [], []
        for outputs in outputs_list:
            logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
            for i, t in enumerate(targets):
                q = logits.shape[1] if batch_queries_num is None else batch_queries_num[i]
                cost = self._cost(logits[i, :q], boxes[i, :q], t["labels"], t["boxes"])
                costs.append(cost.flatten())
                shapes.append(tuple(cost.shape))
        flat = torch.nan_to_num(torch.cat(costs), nan=1.0).cpu()  # the one device-to-host copy
        mats = [m.view(shape).numpy() for m, shape in zip(flat.split([q * n for q, n in shapes]), shapes)]
        pairs = list(_ASSIGN_POOL.map(linear_sum_assignment, mats))

        lengths = [len(i) for i, _ in pairs]
        packed = torch.from_numpy(np.concatenate([np.stack([i, j]) for i, j in pairs], axis=1)).to(device)
        rows, cols = packed[0].split(lengths), packed[1].split(lengths)
        indices = list(zip(rows, cols))
        b = len(targets)
        return [indices[k * b : (k + 1) * b] for k in range(len(outputs_list))]
