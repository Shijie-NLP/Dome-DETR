"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

Rank & Sort Loss (Oksuz et al., ICCV 2021), the ranking-based classification loss: the positives
are to rank above every negative (the ranking error, a soft count of the negatives above each
positive over its rank) and among themselves in the order of their localization quality (the
sorting error, how far each positive's current sorting is from the one its quality targets
prescribe). AP is a ranking measure, and a score head trained per sample (VFL: score = IoU) gets
the order only as a by-product; this trains the order itself.

The loss is not a function whose gradient autograd could take: it is defined through its
"identity update" gradients, which the forward pass computes alongside the errors (a positive
pushed down by its errors, the negatives and the mis-sorted positives above it pushed up in
proportion to how far above they are). ``RankSort.apply`` is a ``torch.autograd.Function`` that
returns the errors and backpropagates those gradients. The original loops over the positives;
this is the same computation on ``[P, P]`` and ``[P, N]`` relation matrices.
"""

import torch
from torch import Tensor

__all__ = ["RankSort", "rank_sort_loss"]


def _step(x: Tensor, delta: float) -> Tensor:
    """The smoothed unit step H(x): 0 below -delta, 1 above delta, linear between (the hard step at delta 0)."""
    if delta > 0:
        return (x / (2 * delta) + 0.5).clamp(0, 1)
    return (x >= 0).to(x.dtype)


class RankSort(torch.autograd.Function):
    """
    Args:
        logits (Tensor): ``[N]`` the classification logits of every (candidate, class) entry.
        targets (Tensor): ``[N]`` their targets: the localization quality in (0, 1] of the
            positives, 0 for the negatives.
        delta (float): half-width of the smoothed step; 0.5 in the paper.

    Returns:
        ``(ranking_error, sorting_error)``, each the mean over the positives; their sum is the loss.
        Without a positive both are 0 and nothing flows back.
    """

    @staticmethod
    def forward(ctx, logits: Tensor, targets: Tensor, delta: float = 0.5, eps: float = 1e-10):
        logits, targets = logits.detach().float(), targets.detach().float()
        grad = torch.zeros_like(logits)
        positive = targets > 0
        num_pos = int(positive.sum())
        zero = logits.new_zeros(())
        if num_pos == 0:
            ctx.save_for_backward(grad)
            return zero, zero
        s, y = logits[positive], targets[positive]  # [P]
        # only the negatives that can rank above a positive matter; the rest have zero relations
        relevant = (~positive) & (logits >= s.min() - delta)
        b = logits[relevant]  # [N]

        fg_rel = _step(s[None, :] - s[:, None], delta)  # [P, P]: how far positive j ranks above positive i
        bg_rel = _step(b[None, :] - s[:, None], delta)  # [P, N]: how far negative k ranks above positive i
        rank_pos = fg_rel.sum(1)  # includes i itself (H(0) = 0.5 with delta > 0, as in the original)
        fp_num = bg_rel.sum(1)
        ranking_error = fp_num / (rank_pos + fp_num)

        one_minus_y = 1 - y
        current_sorting = (fg_rel * one_minus_y[None, :]).sum(1) / rank_pos
        iou_rel = y[None, :] >= y[:, None]  # [P, P]: positives at least as good as i belong above it
        target_sorted = iou_rel.to(fg_rel.dtype) * fg_rel
        target_sorting = (target_sorted * one_minus_y[None, :]).sum(1) / target_sorted.sum(1).clamp_min(eps)
        sorting_error = current_sorting - target_sorting

        # identity-update gradients: each positive's errors go down on it, and up on what ranks above it
        bg_grad = (bg_rel * (ranking_error / fp_num.clamp_min(eps) * (fp_num > eps))[:, None]).sum(0)  # [N]
        missorted = (~iou_rel).to(fg_rel.dtype) * fg_rel  # [P, P]: worse positives ranked above i
        denom = missorted.sum(1)
        fg_grad = (missorted * (sorting_error / denom.clamp_min(eps) * (denom > eps))[:, None]).sum(0)  # [P]
        fg_grad = fg_grad - (ranking_error + sorting_error)

        grad[positive] = fg_grad / num_pos
        grad[relevant] = bg_grad / num_pos
        ctx.save_for_backward(grad)
        return ranking_error.mean(), sorting_error.mean()

    @staticmethod
    def backward(ctx, grad_ranking, grad_sorting):
        (grad,) = ctx.saved_tensors
        # the two errors share one gradient (the original scales it by the ranking error's only)
        return grad * grad_ranking, None, None, None


def rank_sort_loss(logits: Tensor, targets: Tensor, delta: float = 0.5) -> Tensor:
    """The Rank & Sort loss of flat ``logits`` against flat ``targets`` (see ``RankSort``): ranking error + sorting error."""
    ranking_error, sorting_error = RankSort.apply(logits, targets, delta)
    return ranking_error + sorting_error
