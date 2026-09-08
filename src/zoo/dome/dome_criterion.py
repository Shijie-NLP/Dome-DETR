"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

from contextlib import contextmanager
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor

from ...core import register
from ...misc import dist_utils
from ...misc.box_ops import (
    box_cxcywh_to_xyxy,
    elementwise_box_iou,
    elementwise_generalized_box_iou,
    gaussian_box_similarity,
)
from .fdr import bbox2distance
from .matcher import padded_targets, topk_matching

__all__ = ["DomeCriterion"]


class Targets(NamedTuple):
    """A batch's ground truths padded to ``[B, M]``: see ``matcher.padded_targets``."""

    labels: Tensor  # [B, M]
    boxes: Tensor  # [B, M, 4]
    gt_valid: Tensor  # [B, M]
    q_valid: Tensor  # [B, Q], the real (unpadded) queries of every image


class Pairs(NamedTuple):
    """
    The matched pairs of a set stack as flat index tensors, sorted by set: pair ``k`` matches
    query ``query_idx[k]`` of image ``batch_idx[k]`` in set ``set_idx[k]`` to that image's
    ground truth ``target_idx[k]`` (an index into the padded targets). ``counts`` is the number
    of pairs of every set, on the host.
    """

    set_idx: Tensor
    batch_idx: Tensor
    query_idx: Tensor
    target_idx: Tensor
    counts: list[int]

    @classmethod
    def from_lists(cls, indices_lists, device):
        """From one per-image list of ``(pred_idx, target_idx)`` per set, as the matcher returns them."""
        lengths = [[src.shape[0] for src, _ in indices] for indices in indices_lists]
        counts = [sum(lens) for lens in lengths]
        set_idx = torch.arange(len(indices_lists)).repeat_interleave(torch.tensor(counts, dtype=torch.long))
        batch_idx = torch.cat(
            [torch.arange(len(lens)).repeat_interleave(torch.tensor(lens, dtype=torch.long)) for lens in lengths]
        )
        query_idx = torch.cat([src for indices in indices_lists for src, _ in indices])
        target_idx = torch.cat([tgt for indices in indices_lists for _, tgt in indices])
        return cls(
            set_idx.to(device, non_blocking=True),
            batch_idx.to(device, non_blocking=True),
            query_idx,
            target_idx,
            counts,
        )

    @classmethod
    def shared(cls, indices, num_sets, device):
        """The same per-image matching for every one of ``num_sets`` sets."""
        one = cls.from_lists([indices], device)
        k = one.counts[0]
        set_idx = torch.arange(num_sets, device=device).repeat_interleave(k)
        return cls(
            set_idx,
            one.batch_idx.repeat(num_sets),
            one.query_idx.repeat(num_sets),
            one.target_idx.repeat(num_sets),
            [k] * num_sets,
        )

    def first_sets(self, n):
        """The pairs of the first ``n`` sets."""
        k = sum(self.counts[:n])
        return Pairs(self.set_idx[:k], self.batch_idx[:k], self.query_idx[:k], self.target_idx[:k], self.counts[:n])


class SetStack(NamedTuple):
    """
    Prediction sets of the same shape stacked along a leading set dimension, so that one loss
    computation covers them all: ``logits [S, B, Q, C]`` and ``boxes [S, B, Q, 4]`` of every
    set; ``corners [S', B, Q, 4 * (reg_max + 1)]`` and ``refs [S', B, Q, 4]`` of the first ``S'``
    sets, which carry FDR's edge distributions (``None`` when none does); the distillation
    teacher of those sets, if any, with ``has_teacher`` / ``is_teacher`` per set (a set is its own
    teacher in the denoising stack; its distillation loss is zero); the loss suffix per set;
    ``q_valid`` (``None``: every query is real); and whether the sets are denoising ones.
    """

    logits: Tensor
    boxes: Tensor
    corners: Tensor | None
    refs: Tensor | None
    teacher_corners: Tensor | None
    teacher_logits: Tensor | None
    has_teacher: list[bool]
    is_teacher: list[bool]
    suffixes: list[str]
    q_valid: Tensor | None
    is_dn: bool

    @classmethod
    def build(cls, sets, suffixes, q_valid, is_dn=False):
        logits = torch.stack([s["pred_logits"] for s in sets])
        boxes = torch.stack([s["pred_boxes"] for s in sets])
        with_corners = [s for s in sets if "pred_corners" in s]
        assert with_corners == sets[: len(with_corners)], "the sets with edge distributions come first"
        corners = refs = teacher_corners = teacher_logits = None
        has_teacher, is_teacher = [], []
        if with_corners:
            corners = torch.stack([s["pred_corners"] for s in with_corners])
            refs = torch.stack([s["ref_points"] for s in with_corners]).detach()
            teacher_corners = next(
                (s["teacher_corners"] for s in with_corners if s.get("teacher_corners") is not None), None
            )
            teacher_logits = next(
                (s["teacher_logits"] for s in with_corners if s.get("teacher_logits") is not None), None
            )
            has_teacher = [s.get("teacher_corners") is not None for s in with_corners]
            is_teacher = [
                teacher_corners is not None and s["pred_corners"].data_ptr() == teacher_corners.data_ptr()
                for s in with_corners
            ]
        return cls(
            logits,
            boxes,
            corners,
            refs,
            teacher_corners,
            teacher_logits,
            has_teacher,
            is_teacher,
            suffixes,
            q_valid,
            is_dn,
        )

    @property
    def num_sets(self):
        return self.logits.shape[0]

    @property
    def num_corner_sets(self):
        return 0 if self.corners is None else self.corners.shape[0]


def _per_set_sum(values: Tensor, counts: list[int]) -> Tensor:
    """The sum of ``values`` (one per pair, sorted by set) within each set, ``[S]``."""
    if len(set(counts)) == 1:
        return values.view(len(counts), -1).sum(1)
    return torch.stack([v.sum() for v in values.split(counts)])


@register()
class DomeCriterion(nn.Module):
    """
    The Dome-DETR training loss: D-FINE's set-prediction losses on every prediction set the
    decoder returns, plus DeFE's density-map and count losses.

    The decoder output carries, besides the last layer's predictions, ``aux_outputs`` (the other
    layers), ``pre_outputs`` (the first layer's plain boxes), ``enc_aux_outputs`` (the encoder
    tokens picked as queries), and their denoising twins ``dn_outputs`` / ``dn_pre_outputs``. Each
    set is matched to the targets and scored with the same ``losses`` (the encoder sets with
    ``enc_losses`` when given), and the weighted terms are returned with a suffix naming the set
    (``_aux_0``, ``_pre``, ``_enc_0``, ``_dn_0``, ...). Padded queries (``batch_queries_num``)
    are never matched and count in no loss.

    Sets of one kind (the decoder layers with the pre-outputs, the encoder sets, the denoising
    sets) are stacked (``SetStack``) and their matches flattened (``Pairs``), so every loss runs
    once over a whole stack and reduces per set.

    Args:
        matcher: the Hungarian matcher (injected from the config).
        weight_dict: weight per loss term; terms not listed are dropped.
        losses: which of ``vfl`` / ``focal`` / ``mal`` / ``obj`` (classification), ``boxes`` (L1 +
            GIoU) and ``local`` (FDR's fine-grained localization and distillation losses) to
            compute.
        enc_losses: the losses of the encoder sets instead (``None``: ``losses``); e.g.
            ``['obj', 'boxes']`` trains the encoder's class logits as plain 0/1 objectness.
        enc_matching: how the encoder sets are matched: ``hungarian`` (D-FINE, one query per
            ground truth) or ``topk`` (``matcher.topk_matching``: every ground truth's
            ``enc_topk`` most similar queries are its positives, a query serving one ground truth
            at most). The encoder's queries are candidates, not detections, so one-to-one is not
            needed there, and several passing tokens per object make its recall robust.
        enc_topk: the queries per ground truth of ``topk``.
        enc_in_uni_set: whether the encoder sets' matches join the union matching (D-FINE: yes).
            With ``topk`` they should not, or the decoder's box losses would turn one-to-many.
            The encoder sets' box losses then use their own matches.
        alpha, gamma: the focal parameters of the classification losses.
        reg_max: the FDR bin count of the decoder.
        quality: the localization quality of a matched pair, the VFL / MAL target score and the
            FGL weight: ``iou`` (D-FINE), ``giou`` (clamped at 0), ``nwd`` (the normalized Gaussian
            Wasserstein distance, ``exp(-W2 / nwd_c)``) or ``gaussian`` (one minus the Hellinger
            distance between the boxes as Gaussians: parameter-free, scale-invariant, smooth,
            and defined for boxes that do not overlap; about three times less sensitive to a
            small offset than IoU, which is what tiny objects need).
        nwd_c: the ``nwd`` constant, in normalized units (0.016 is 12.8 px of an 800 px image).
        boxes_weight_format: ``None``, ``iou`` or ``giou``: weight the GIoU loss and the VFL / MAL
            targets by the matched pairs' (G)IoU instead of ``quality``.
        defe_density_map_weight, density_recall_penalty: the density-map loss weight, and how
            much harder under-estimation of populated cells is penalised.
        mal_alpha: the negative weight of the MAL loss (``None``: 1).
        use_uni_set: match the box and localization losses against the union of the matches of
            every prediction set (D-FINE's 'go' indices) rather than each set's own.
        obj_quality_weight: in ``loss_obj``, share the positive half among the positives in
            proportion to their ``quality`` (the pair's Gaussian similarity with ``gaussian``)
            instead of equally: the targets stay 1, so every positive is still pushed past the
            boundary (recall), but a positive whose box is well off its ground truth pushes less.
        obj_pos_weight: in the 0/1 classification loss (``loss_obj``, a class-balanced BCE over
            every (query, class) entry, the matched queries' ground-truth class positive),
            how much more the positive half weighs than the negative half; 1 is the balanced
            decision boundary at logit 0.
    """

    __share__ = ["num_classes"]
    __inject__ = ["matcher"]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        quality="iou",
        nwd_c=0.016,
        boxes_weight_format=None,
        defe_density_map_weight=4,
        density_recall_penalty=0.3,
        mal_alpha=None,
        use_uni_set=True,
        enc_losses=None,
        enc_matching="hungarian",
        enc_topk=3,
        enc_in_uni_set=True,
        obj_quality_weight=False,
        obj_pos_weight=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.enc_losses = enc_losses
        assert enc_matching in ("hungarian", "topk"), enc_matching
        assert enc_topk >= 1
        self.enc_matching = enc_matching
        self.enc_topk = enc_topk
        self.enc_in_uni_set = enc_in_uni_set
        assert quality in ("iou", "giou", "nwd", "gaussian"), quality
        self.quality = quality
        self.nwd_c = nwd_c
        self.boxes_weight_format = boxes_weight_format
        self.alpha = alpha
        self.gamma = gamma
        self.reg_max = reg_max
        self.defe_density_map_weight = defe_density_map_weight
        self.density_recall_penalty = density_recall_penalty
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set
        self.obj_quality_weight = obj_quality_weight
        self.obj_pos_weight = obj_pos_weight
        self._clear_cache()

    def _clear_cache(self):
        # per-forward caches: the DDF normalisers of the decoder stack, reused by the denoising
        # stack, and the matched pairs' boxes and quality, gathered once per (stack, pairs)
        self.num_pos, self.num_neg = None, None
        self.matched = {}

    # ------------------------------------------------------------------ matched pairs

    def _matched(self, stack: SetStack, pairs: Pairs, targets: Targets):
        """The matched predictions' boxes ``[K, 4]``, the target boxes they are matched to, and their quality (detached)."""
        key = (id(stack), id(pairs))
        if key not in self.matched:
            src_boxes = stack.boxes[pairs.set_idx, pairs.batch_idx, pairs.query_idx]
            target_boxes = targets.boxes[pairs.batch_idx, pairs.target_idx]
            with torch.no_grad():
                quality = self._matched_quality(src_boxes, target_boxes)
            self.matched[key] = (src_boxes, target_boxes, quality)
        return self.matched[key]

    def _matched_quality(self, src_boxes, target_boxes):
        """The localization quality of matched pairs of cxcywh boxes, ``[K]`` in [0, 1], by ``quality``."""
        if self.quality == "iou":
            return elementwise_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))[0]
        if self.quality == "giou":
            giou = elementwise_generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            return giou.clamp(min=0)
        if self.quality == "nwd":
            delta = src_boxes - target_boxes
            w2 = torch.cat([delta[:, :2], delta[:, 2:] / 2], dim=-1).norm(dim=-1)  # in (cx, cy, w/2, h/2)
            return torch.exp(-w2 / self.nwd_c)
        return gaussian_box_similarity(src_boxes, target_boxes)

    def _class_targets(self, stack: SetStack, pairs: Pairs, targets: Targets):
        """Per (set, query) target class ``[S, B, Q]`` (``num_classes`` = background) and its one-hot over the real classes."""
        classes = torch.full(stack.logits.shape[:3], self.num_classes, dtype=torch.int64, device=stack.logits.device)
        classes[pairs.set_idx, pairs.batch_idx, pairs.query_idx] = targets.labels[pairs.batch_idx, pairs.target_idx]
        one_hot = F.one_hot(classes, num_classes=self.num_classes + 1)[..., :-1]
        return classes, one_hot

    def _reduce_query_loss(self, loss: Tensor, stack: SetStack, num_boxes):
        """Sum a ``[S, B, Q, C]`` per-query loss over each set, ignoring the padded queries, normalized by ``num_boxes``."""
        if stack.q_valid is not None:
            loss = loss * stack.q_valid[None, :, :, None]
        return loss.sum((1, 2, 3)) / num_boxes

    # ------------------------------------------------------------------ classification losses

    def loss_labels_focal(self, stack, pairs, num_boxes, targets, **kwargs):
        _, target = self._class_targets(stack, pairs, targets)
        target = target.to(stack.logits.dtype)  # the one-hot is int64; BCE needs a float target
        loss = torchvision.ops.sigmoid_focal_loss(stack.logits, target, self.alpha, self.gamma, reduction="none")
        return {"loss_focal": self._reduce_query_loss(loss, stack, num_boxes)}

    def _iou_aware_targets(self, stack, pairs, targets, values):
        """Shared by VFL and MAL: the one-hot targets with the matched quality (or ``values``) as the positive score."""
        logits = stack.logits
        _, _, quality = self._matched(stack, pairs, targets)
        ious = quality if values is None else values
        classes, target = self._class_targets(stack, pairs, targets)
        target_score = torch.zeros_like(classes, dtype=logits.dtype)
        target_score[pairs.set_idx, pairs.batch_idx, pairs.query_idx] = ious.to(target_score.dtype)
        return logits, target, target_score.unsqueeze(-1) * target

    def loss_labels_vfl(self, stack, pairs, num_boxes, targets, values=None, **kwargs):
        src_logits, target, target_score = self._iou_aware_targets(stack, pairs, targets, values)
        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        return {"loss_vfl": self._reduce_query_loss(loss, stack, num_boxes)}

    def loss_labels_mal(self, stack, pairs, num_boxes, targets, values=None, **kwargs):
        src_logits, target, target_score = self._iou_aware_targets(stack, pairs, targets, values)
        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        neg_weight = 1.0 if self.mal_alpha is None else self.mal_alpha
        weight = neg_weight * pred_score.pow(self.gamma) * (1 - target) + target
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        return {"loss_mal": self._reduce_query_loss(loss, stack, num_boxes)}

    def loss_obj(self, stack, pairs, num_boxes, targets, **kwargs):
        """
        Plain 0/1 classification, no IoU-aware target: a class-balanced BCE over every (query,
        class) entry of each set's ``pred_logits [B, Q, C]``. The matched queries are positive on
        their ground-truth class (class 0 when ``C`` is 1), every other entry negative; each half
        is normalized to weight 1/2 (the positives' shares equal, or with ``obj_quality_weight``
        in proportion to their ``quality``) and the positive half scaled by ``obj_pos_weight``, so
        the decision boundary is logit 0. Padded queries are left out.
        """
        logits = stack.logits.float()  # the weights are built in fp32 under autocast too
        s, b, q = pairs.set_idx, pairs.batch_idx, pairs.query_idx
        cls = targets.labels[b, pairs.target_idx] if logits.shape[-1] > 1 else torch.zeros_like(q)
        target = torch.zeros_like(logits)
        target[s, b, q, cls] = 1.0
        share = torch.zeros(logits.shape[:3], device=logits.device)  # each positive query's share of the positive half
        share[s, b, q] = 1.0
        if self.obj_quality_weight:
            share[s, b, q] = self._matched(stack, pairs, targets)[2].to(share.dtype)
        keep = torch.ones_like(logits, dtype=torch.bool)
        if stack.q_valid is not None:
            keep &= stack.q_valid[None, :, :, None]
        pos, neg = (target > 0) & keep, (target == 0) & keep
        share = share[..., None] * pos  # [S, B, Q, C], the positives' shares
        pos_weight = 0.5 * self.obj_pos_weight * share / share.sum((1, 2, 3)).clamp(min=1e-6)[:, None, None, None]
        neg_weight = 0.5 / neg.sum((1, 2, 3)).clamp(min=1)[:, None, None, None]
        weight = pos_weight * pos + neg_weight * neg
        loss = F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="none")
        return {"loss_obj": loss.sum((1, 2, 3))}

    # ------------------------------------------------------------------ box losses

    def loss_boxes(self, stack, pairs, num_boxes, targets, boxes_weight=None, **kwargs):
        """L1 and GIoU losses of the matched pairs (boxes are normalized cxcywh)."""
        src_boxes, target_boxes, _ = self._matched(stack, pairs, targets)
        loss_bbox = _per_set_sum((src_boxes - target_boxes).abs().sum(-1), pairs.counts) / num_boxes
        loss_giou = 1 - elementwise_generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        if boxes_weight is not None:
            loss_giou = loss_giou * boxes_weight
        return {"loss_bbox": loss_bbox, "loss_giou": _per_set_sum(loss_giou, pairs.counts) / num_boxes}

    def loss_local(self, stack, pairs, num_boxes, targets, fdr, T=5, **kwargs):  # noqa: N803
        """
        FDR's Fine-Grained Localization (FGL) loss on the matched pairs' edge distributions of
        the sets that have them, and, for those with a distillation teacher, the Decoupled
        Distillation Focal (DDF) loss towards it. ``pairs`` are those sets' pairs.
        """
        if stack.corners is None:
            return {}
        s, b, q = pairs.set_idx, pairs.batch_idx, pairs.query_idx
        src_boxes, target_boxes, quality = self._matched(stack, pairs, targets)
        pred_corners = stack.corners[s, b, q].reshape(-1, self.reg_max + 1)  # [K * 4, bins]
        with torch.no_grad():
            target_corners, weight_right, weight_left = bbox2distance(
                stack.refs[s, b, q],
                box_cxcywh_to_xyxy(target_boxes),
                self.reg_max,
                fdr["reg_scale"],
                fdr["up"],
                min_unit=fdr["min_unit"],
            )
        weight_targets = quality.unsqueeze(-1).expand(-1, 4).reshape(-1)
        fgl = self.unimodal_distribution_focal_loss(
            pred_corners, target_corners, weight_right, weight_left, weight_targets
        )
        losses = {"loss_fgl": _per_set_sum(fgl, [c * 4 for c in pairs.counts]) / num_boxes}
        if stack.teacher_corners is not None:
            losses["loss_ddf"] = self._loss_ddf(stack, pairs, quality, T)
        return losses

    def _loss_ddf(self, stack: SetStack, pairs: Pairs, quality: Tensor, T):  # noqa: N803
        """KL distillation of every query's edge distributions towards the teacher's, at temperature ``T``, ``[S']``."""
        num_sets, b, q = stack.corners.shape[:3]
        pred_corners = stack.corners.reshape(num_sets, b, q, 4, -1)
        target_corners = stack.teacher_corners.detach().reshape(b, q, 4, -1)

        # matched queries are weighted by their quality, the others by the teacher's confidence
        weight = stack.teacher_logits.sigmoid().max(dim=-1)[0].expand(num_sets, -1, -1).clone()
        weight[pairs.set_idx, pairs.batch_idx, pairs.query_idx] = quality.to(weight.dtype)
        mask = torch.zeros(num_sets, b, q, dtype=torch.bool, device=weight.device)
        mask[pairs.set_idx, pairs.batch_idx, pairs.query_idx] = True
        weight, mask = weight[..., None].expand(-1, -1, -1, 4).detach(), mask[..., None].expand(-1, -1, -1, 4)

        kl = nn.KLDivLoss(reduction="none")(
            F.log_softmax(pred_corners / T, dim=-1), F.softmax(target_corners / T, dim=-1)
        ).sum(-1)
        loss_match_local = weight * (T**2) * kl  # [S', B, Q, 4]

        if not stack.is_dn:
            # balance the matched and unmatched halves; sqrt-scaled so that the GPU batch size does not matter
            batch_scale = 8 / b
            self.num_pos = (mask.sum((1, 2, 3)) * batch_scale) ** 0.5
            self.num_neg = ((~mask).sum((1, 2, 3)) * batch_scale) ** 0.5
        # the halves' means, 0 for an empty half, without reading the masks on the host
        loss_pos = (loss_match_local * mask).sum((1, 2, 3)) / mask.sum((1, 2, 3)).clamp(min=1)
        loss_neg = (loss_match_local * ~mask).sum((1, 2, 3)) / (~mask).sum((1, 2, 3)).clamp(min=1)
        loss = (loss_pos * self.num_pos + loss_neg * self.num_neg) / (self.num_pos + self.num_neg)
        # a set that is its own teacher (the last denoising layer) distils nothing
        own = torch.tensor([0.0 if t else 1.0 for t in stack.is_teacher], device=loss.device)
        return loss * own

    @staticmethod
    def unimodal_distribution_focal_loss(pred, label, weight_right, weight_left, weight=None):
        """Cross-entropy against the two bins around each target position, weighted by their distance to it, per element."""
        dis_left = label.long()
        dis_right = dis_left + 1
        loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(-1)
        loss = loss + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)
        if weight is not None:
            loss = loss * weight.float()
        return loss

    # ------------------------------------------------------------------ DeFE losses

    def loss_defe(self, defe, targets):
        """
        The density-map loss (a squared error weighted up where the map under-estimates populated
        cells) and, when the decoder is ``DomeTransformer`` (it writes its query budget into
        ``defe``), the count regression loss: a squared error on the object count normalized to
        that budget, doubled when the prediction falls short. With ``DFINETransformer`` there is
        no budget to normalize to and the count head is left untrained.
        """
        density_map, gt_density_map = defe["defe_feature"], defe["gt_density_map"]
        under = (density_map < gt_density_map).float()
        penalty = 1 + self.density_recall_penalty * gt_density_map * under
        defe_density_loss = (penalty * (density_map - gt_density_map) ** 2).mean() * self.defe_density_map_weight
        losses = {"defe_density_loss": defe_density_loss}

        if "min_num_select" in defe:
            min_n, max_n = defe["min_num_select"], defe["max_num_select"]
            reg_value = defe["reg_value"]
            # NOTE: kept exactly as trained upstream. The normalized count is cast to int64, which
            # truncates every target below max_num_select to 0, and reg_value [B, 1] broadcasts
            # against the [B] targets to a [B, B] difference.
            counts = [min(max(len(t["labels"]), min_n), max_n) for t in targets]
            reg_targets = torch.tensor(
                [(c - min_n) / (max_n - min_n) for c in counts], dtype=torch.int64, device=reg_value.device
            )
            diff = reg_value - reg_targets
            penalty = torch.where(diff < 0, 2.0, 1.0)
            losses["defe_reg_loss"] = (penalty * diff**2).mean()
        return losses

    # ------------------------------------------------------------------ assembling

    def get_loss(self, loss, stack, pairs, num_boxes, targets, **kwargs):
        loss_map = {
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "mal": self.loss_labels_mal,
            "obj": self.loss_obj,
            "local": self.loss_local,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](stack, pairs, num_boxes, targets, **kwargs)

    def get_loss_meta_info(self, loss, stack, pairs, targets):
        """With ``boxes_weight_format``, the matched pairs' (G)IoU as the weight / target score of a loss."""
        if self.boxes_weight_format is None:
            return {}
        src_boxes, target_boxes, _ = self._matched(stack, pairs, targets)
        src_xyxy, tgt_xyxy = box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
        if self.boxes_weight_format == "iou":
            iou = elementwise_box_iou(src_xyxy, tgt_xyxy)[0]
        elif self.boxes_weight_format == "giou":
            iou = elementwise_generalized_box_iou(src_xyxy, tgt_xyxy)
        else:
            raise ValueError(f"unknown boxes_weight_format {self.boxes_weight_format!r}")
        if loss == "boxes":
            return {"boxes_weight": iou}
        if loss in ("vfl", "mal"):
            return {"values": iou}
        return {}

    def _stack_losses(self, stack, targets, losses, own, shared, num_boxes, num_shared, uni_losses, fdr=None):
        """
        Every loss in ``losses`` over a stack, weighted and suffixed per set. Losses named in
        ``uni_losses`` use the union matching ``shared`` and its pair count ``num_shared``
        instead of the sets' ``own`` pairs and ``num_boxes`` (a float, or one per set).
        """
        result = {}
        local_pairs = {}
        for loss in losses:
            pairs, nb = (shared, num_shared) if (self.use_uni_set and loss in uni_losses) else (own, num_boxes)
            if loss == "local":  # the sets with edge distributions come first
                if id(pairs) not in local_pairs:
                    local_pairs[id(pairs)] = pairs.first_sets(stack.num_corner_sets)
                pairs = local_pairs[id(pairs)]
            meta = self.get_loss_meta_info(loss, stack, pairs, targets)
            per_set = self.get_loss(loss, stack, pairs, nb, targets, fdr=fdr, **meta)
            for k, v in per_set.items():
                if k not in self.weight_dict:
                    continue
                for s, suffix in enumerate(stack.suffixes[: v.shape[0]]):
                    if k == "loss_ddf" and not stack.has_teacher[s]:
                        continue
                    result[k + suffix] = v[s] * self.weight_dict[k]
        return result

    @staticmethod
    def _average_over_ranks(count, device) -> float:
        """A count averaged over the distributed ranks, at least 1."""
        if not dist_utils.is_dist_available_and_initialized():
            return float(max(count, 1))
        count = torch.as_tensor([count], dtype=torch.float, device=device)
        torch.distributed.all_reduce(count)
        return torch.clamp(count / dist_utils.get_world_size(), min=1).item()

    @staticmethod
    def _union_indices(indices_list, num_queries, num_targets):
        """
        D-FINE's 'go' matching: the union of the matches of several prediction sets
        (``indices_list``, one per-image list each), keeping for every query the target it was
        matched to most often (the lowest index on a tie). Three host syncs.
        """
        b, m = len(num_targets), max(num_targets)
        device = indices_list[0][0][0].device
        if m == 0:
            empty = torch.zeros(0, dtype=torch.long, device=device)
            return [(empty, empty) for _ in range(b)]
        # every match as one key (image, query, target); the number of sets a key appears in
        key = torch.cat([(i * num_queries + src) * m + tgt for ind in indices_list for i, (src, tgt) in enumerate(ind)])
        key, count = torch.unique(key, return_counts=True)  # sorted: image, query, then target
        query, target = key // m, key % m  # query numbered across the batch
        best = torch.zeros((b * num_queries,), dtype=count.dtype, device=device)
        best.scatter_reduce_(0, query, count, "amax")
        top = count == best[query]
        first = torch.full((b * num_queries,), m, dtype=torch.long, device=device)
        first.scatter_reduce_(0, query[top], target[top], "amin")  # the lowest target among the ties
        keep = top & (target == first[query])
        query, target = query[keep], target[keep]
        per_image = torch.bincount(query // num_queries, minlength=b).tolist()
        return list(zip((query % num_queries).split(per_image), target.split(per_image)))

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """Every denoising query is matched to the ground truth it was made from, group after group."""
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        device = targets[0]["labels"].device
        dn_match_indices = []
        for i, t in enumerate(targets):
            num_gt = len(t["labels"])
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device).tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                empty = torch.zeros(0, dtype=torch.int64, device=device)
                dn_match_indices.append((empty, empty))
        return dn_match_indices

    @contextmanager
    def _class_agnostic(self, targets: Targets):
        """Score against a single class: every label becomes 0 and ``num_classes`` is 1 for the duration."""
        num_classes = self.num_classes
        self.num_classes = 1
        try:
            yield targets._replace(labels=torch.zeros_like(targets.labels))
        finally:
            self.num_classes = num_classes

    def forward(self, outputs, targets, **kwargs):
        assert "aux_outputs" in outputs, "DomeCriterion needs the decoder's auxiliary outputs (aux_loss: True)"
        device = outputs["pred_logits"].device
        batch_queries_num = outputs.get("batch_queries_num")
        num_queries = outputs["pred_logits"].shape[1]
        self._clear_cache()
        padded = Targets(*padded_targets(targets, num_queries, batch_queries_num))
        fdr = {"up": outputs.get("up"), "reg_scale": outputs.get("reg_scale"), "min_unit": outputs.get("fdr_min_unit")}

        # match every prediction set in one go, and build the union matching for the box losses
        hungarian_enc = self.enc_matching == "hungarian"
        sets = [outputs, *outputs["aux_outputs"], outputs["pre_outputs"]]
        suffixes = ["", *(f"_aux_{i}" for i in range(len(outputs["aux_outputs"]))), "_pre"]
        enc_sets = outputs["enc_aux_outputs"]
        matched = self.matcher.match_sets(
            sets + (enc_sets if hungarian_enc else []), targets, batch_queries_num=batch_queries_num
        )
        indices = matched[: len(sets)]
        if hungarian_enc:
            indices_enc = matched[len(sets) :]
        else:
            indices_enc = [topk_matching(o, targets, self.enc_topk, batch_queries_num) for o in enc_sets]
        num_targets = [len(t["labels"]) for t in targets]
        indices_go = self._union_indices(
            indices + (indices_enc if self.enc_in_uni_set else []), num_queries, num_targets
        )
        num_go = self._average_over_ranks(sum(len(x[0]) for x in indices_go), device)
        num_boxes = self._average_over_ranks(sum(num_targets), device)

        # the decoder's sets: their own matches for the classification losses, the union for the boxes
        stack = SetStack.build(sets, suffixes, padded.q_valid)
        own = Pairs.from_lists(indices, device)
        shared = Pairs.shared(indices_go, stack.num_sets, device)
        losses = self._stack_losses(stack, padded, self.losses, own, shared, num_boxes, num_go, ("boxes", "local"), fdr)

        # the encoder sets: their own losses, and outside the union their own matches and pair counts
        enc_stack = SetStack.build(enc_sets, [f"_enc_{i}" for i in range(len(enc_sets))], padded.q_valid)
        enc_own = Pairs.from_lists(indices_enc, device)
        enc_losses = self.losses if self.enc_losses is None else self.enc_losses
        if self.enc_in_uni_set:
            enc_args = (enc_own, Pairs.shared(indices_go, enc_stack.num_sets, device), num_boxes, num_go, ("boxes",))
        else:
            pairs = [self._average_over_ranks(sum(len(x[0]) for x in ind), device) for ind in indices_enc]
            enc_args = (enc_own, enc_own, torch.tensor(pairs, device=device), None, ())
        if outputs["enc_meta"]["class_agnostic"]:
            with self._class_agnostic(padded) as enc_targets:
                losses.update(self._stack_losses(enc_stack, enc_targets, enc_losses, *enc_args, fdr))
        else:
            losses.update(self._stack_losses(enc_stack, padded, enc_losses, *enc_args, fdr))

        if "dn_outputs" in outputs:
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_sets = [*outputs["dn_outputs"], outputs["dn_pre_outputs"]]
            dn_suffixes = [*(f"_dn_{i}" for i in range(len(outputs["dn_outputs"]))), "_dn_pre"]
            dn_stack = SetStack.build(dn_sets, dn_suffixes, None, is_dn=True)
            dn_pairs = Pairs.shared(indices_dn, dn_stack.num_sets, device)
            dn_num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]
            losses.update(
                self._stack_losses(dn_stack, padded, self.losses, dn_pairs, dn_pairs, dn_num_boxes, None, (), fdr)
            )

        if "defe" in outputs:
            losses.update(self.loss_defe(outputs["defe"], targets))

        # a NaN term must not take the whole step down with it
        return {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
