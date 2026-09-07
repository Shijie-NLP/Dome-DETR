"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from scipy.optimize import linear_sum_assignment

from ...core import register
from ...misc.box_ops import box_cxcywh_to_xyxy, gaussian_box_similarity, generalized_box_iou

__all__ = ["HungarianMatcher"]


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
            image, each of length ``min(Q, N_i)``.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        logits = outputs["pred_logits"].flatten(0, 1)  # [B * Q, C]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B * Q, 4]
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # the constant 1 of "1 - p[target]" does not change the matching and is left out
        if self.use_focal_loss:
            out_prob = F.sigmoid(logits)[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -logits.softmax(-1)[:, tgt_ids]

        cost = self.cost_class * cost_class + self.cost_bbox * torch.cdist(out_bbox, tgt_bbox, p=1)
        if self.cost_giou:
            cost = cost - self.cost_giou * generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
            )
        if self.cost_gaussian:
            cost = cost + self.cost_gaussian * (1 - gaussian_box_similarity(out_bbox[:, None, :], tgt_bbox[None, :, :]))
        cost = cost.view(bs, num_queries, -1)
        if batch_queries_num is not None:  # padded queries cost more than any real one
            counts = torch.tensor(batch_queries_num).to(cost.device, non_blocking=True)
            pad = torch.arange(num_queries, device=cost.device)[None, :] >= counts[:, None]
            cost = cost.masked_fill(pad[..., None], 1e6)
        cost = torch.nan_to_num(cost.cpu(), nan=1.0)

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]
        return {
            "indices": [
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices
            ]
        }
