"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision

from ...core import register
from ...misc import dist_utils
from ...misc.box_ops import (
    box_cxcywh_to_xyxy,
    elementwise_box_iou,
    elementwise_generalized_box_iou,
    gaussian_box_similarity,
)
from .fdr import bbox2distance

__all__ = ["DomeCriterion"]


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

    Args:
        matcher: the Hungarian matcher (injected from the config).
        weight_dict: weight per loss term; terms not listed are dropped.
        losses: which of ``vfl`` / ``focal`` / ``mal`` / ``obj`` (classification), ``boxes`` (L1 +
            GIoU) and ``local`` (FDR's fine-grained localization and distillation losses) to
            compute.
        enc_losses: the losses of the encoder sets instead (``None``: ``losses``); e.g.
            ``['obj', 'boxes']`` trains the encoder's class logits as plain 0/1 objectness.
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
        obj_pos_weight=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.enc_losses = enc_losses
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
        self.obj_pos_weight = obj_pos_weight
        self._clear_cache()

    def _clear_cache(self):
        # per-forward caches: the FGL targets are the same for every decoder layer (all layers
        # regress from the first layer's reference boxes), and so are the DDF normalisers
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    # ------------------------------------------------------------------ matched pairs

    @staticmethod
    def _get_src_permutation_idx(indices):
        """The (batch, query) index of every matched prediction."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _matched_boxes(self, outputs, targets, indices):
        """The matched predictions' index, their boxes and the target boxes they are matched to (cxcywh)."""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        return idx, src_boxes, target_boxes

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

    def _class_targets(self, src_logits, targets, indices, idx):
        """Per-query target class (``num_classes`` = background) and its one-hot over the real classes."""
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        one_hot = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        return target_classes, one_hot

    @staticmethod
    def _reduce_query_loss(loss, batch_queries_num, num_boxes):
        """Sum a ``[B, Q, C]`` per-query loss, ignoring the padded queries of each image, normalized by ``num_boxes``."""
        if batch_queries_num is not None:
            queries = torch.arange(loss.shape[1], device=loss.device)[None, :]
            valid = queries < torch.as_tensor(batch_queries_num, device=loss.device)[:, None]
            loss = loss * valid.unsqueeze(-1)
        return loss.mean(1).sum() * loss.shape[1] / num_boxes

    # ------------------------------------------------------------------ classification losses

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, batch_queries_num=None, **kwargs):
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        _, target = self._class_targets(src_logits, targets, indices, idx)
        target = target.to(src_logits.dtype)  # the one-hot is int64; BCE needs a float target
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction="none")
        return {"loss_focal": self._reduce_query_loss(loss, batch_queries_num, num_boxes)}

    def _iou_aware_targets(self, outputs, targets, indices, values):
        """Shared by VFL and MAL: the one-hot targets with the matched quality (or ``values``) as the positive score."""
        src_logits = outputs["pred_logits"]
        idx, src_boxes, target_boxes = self._matched_boxes(outputs, targets, indices)
        ious = self._matched_quality(src_boxes, target_boxes).detach() if values is None else values
        target_classes, target = self._class_targets(src_logits, targets, indices, idx)
        target_score = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score[idx] = ious.to(target_score.dtype)
        return src_logits, target, target_score.unsqueeze(-1) * target

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None, batch_queries_num=None, **kwargs):
        src_logits, target, target_score = self._iou_aware_targets(outputs, targets, indices, values)
        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        return {"loss_vfl": self._reduce_query_loss(loss, batch_queries_num, num_boxes)}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None, batch_queries_num=None, **kwargs):
        src_logits, target, target_score = self._iou_aware_targets(outputs, targets, indices, values)
        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        neg_weight = 1.0 if self.mal_alpha is None else self.mal_alpha
        weight = neg_weight * pred_score.pow(self.gamma) * (1 - target) + target
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        return {"loss_mal": self._reduce_query_loss(loss, batch_queries_num, num_boxes)}

    # ------------------------------------------------------------------ box losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None, **kwargs):
        """L1 and GIoU losses of the matched pairs (boxes are normalized cxcywh)."""
        _, src_boxes, target_boxes = self._matched_boxes(outputs, targets, indices)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / num_boxes
        loss_giou = 1 - elementwise_generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        if boxes_weight is not None:
            loss_giou = loss_giou * boxes_weight
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou.sum() / num_boxes}

    def loss_local(self, outputs, targets, indices, num_boxes, T=5, **kwargs):  # noqa: N803
        """
        FDR's Fine-Grained Localization (FGL) loss on the matched pairs' edge distributions, and,
        when the set carries ``teacher_corners`` (the last layer's), the Decoupled Distillation
        Focal (DDF) loss towards them.
        """
        if "pred_corners" not in outputs:
            return {}
        idx, src_boxes, target_boxes = self._matched_boxes(outputs, targets, indices)
        pred_corners = outputs["pred_corners"][idx].reshape(-1, self.reg_max + 1)
        ref_points = outputs["ref_points"][idx].detach()
        is_dn = "is_dn" in outputs

        cache = "fgl_targets_dn" if is_dn else "fgl_targets"
        if getattr(self, cache) is None:
            with torch.no_grad():
                distances = bbox2distance(
                    ref_points,
                    box_cxcywh_to_xyxy(target_boxes),
                    self.reg_max,
                    outputs["reg_scale"],
                    outputs["up"],
                    min_unit=outputs.get("fdr_min_unit"),
                )
            setattr(self, cache, distances)
        target_corners, weight_right, weight_left = getattr(self, cache)

        ious = self._matched_quality(src_boxes, target_boxes)
        weight_targets = ious.unsqueeze(-1).repeat(1, 4).reshape(-1).detach()
        losses = {
            "loss_fgl": self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes
            )
        }
        if "teacher_corners" in outputs:
            losses["loss_ddf"] = self._loss_ddf(outputs, idx, ious, T, is_dn)
        return losses

    def _loss_ddf(self, outputs, idx, ious, T, is_dn):  # noqa: N803
        """KL distillation of every query's edge distributions towards the teacher's, at temperature ``T``."""
        pred_corners = outputs["pred_corners"].reshape(-1, self.reg_max + 1)
        target_corners = outputs["teacher_corners"].reshape(-1, self.reg_max + 1)
        if torch.equal(pred_corners, target_corners):
            return pred_corners.sum() * 0  # the teacher layer itself

        # matched queries are weighted by their IoU, the others by the teacher's confidence
        weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]
        mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
        mask[idx] = True
        mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)
        weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
        weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

        kl = nn.KLDivLoss(reduction="none")(
            F.log_softmax(pred_corners / T, dim=1), F.softmax(target_corners.detach() / T, dim=1)
        ).sum(-1)
        loss_match_local = weight_targets_local * (T**2) * kl

        if not is_dn:
            # balance the matched and unmatched halves; sqrt-scaled so that the GPU batch size does not matter
            batch_scale = 8 / outputs["pred_boxes"].shape[0]
            self.num_pos = (mask.sum() * batch_scale) ** 0.5
            self.num_neg = ((~mask).sum() * batch_scale) ** 0.5
        loss_pos = loss_match_local[mask].mean() if mask.any() else 0
        loss_neg = loss_match_local[~mask].mean() if (~mask).any() else 0
        return (loss_pos * self.num_pos + loss_neg * self.num_neg) / (self.num_pos + self.num_neg)

    @staticmethod
    def unimodal_distribution_focal_loss(pred, label, weight_right, weight_left, weight=None, avg_factor=None):
        """Cross-entropy against the two bins around each target position, weighted by their distance to it."""
        dis_left = label.long()
        dis_right = dis_left + 1
        loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(-1)
        loss = loss + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)
        if weight is not None:
            loss = loss * weight.float()
        return loss.sum() / avg_factor if avg_factor is not None else loss.sum()

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

    def loss_obj(self, outputs, targets, indices, num_boxes, batch_queries_num=None, **kwargs):
        """
        Plain 0/1 classification, no IoU-aware target: a class-balanced BCE over every (query,
        class) entry of ``pred_logits [B, Q, C]``. The matched queries are positive on their
        ground-truth class (class 0 when ``C`` is 1), every other entry negative; each half is
        normalized to weight 1/2 and the positive half scaled by ``obj_pos_weight``, so the
        decision boundary is logit 0. Padded queries are left out.
        """
        logits = outputs["pred_logits"].float()  # the weights are built in fp32 under autocast too
        target = torch.zeros_like(logits)
        for i, (src, tgt) in enumerate(indices):
            cls = targets[i]["labels"][tgt] if logits.shape[-1] > 1 else torch.zeros_like(tgt)
            target[i, src, cls] = 1.0
        keep = torch.ones_like(logits, dtype=torch.bool)
        if batch_queries_num is not None:
            counts = torch.tensor(batch_queries_num).to(logits.device, non_blocking=True)
            keep &= (torch.arange(logits.shape[1], device=logits.device)[None, :] < counts[:, None])[..., None]
        pos, neg = (target > 0) & keep, (target == 0) & keep
        weight = torch.zeros_like(logits)
        weight[pos] = 0.5 * self.obj_pos_weight / pos.sum().clamp(min=1)
        weight[neg] = 0.5 / neg.sum().clamp(min=1)
        return {"loss_obj": F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="sum")}

    # ------------------------------------------------------------------ assembling

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "mal": self.loss_labels_mal,
            "obj": self.loss_obj,
            "local": self.loss_local,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        """With ``boxes_weight_format``, the matched pairs' (G)IoU as the weight / target score of a loss."""
        if self.boxes_weight_format is None:
            return {}
        _, src_boxes, target_boxes = self._matched_boxes(outputs, targets, indices)
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

    def _weighted_losses(
        self, outputs, targets, indices, num_boxes, suffix, uni_losses, shared, batch_queries_num, losses=None
    ):
        """
        Every configured loss (``losses``, default the criterion's) on one prediction set,
        weighted and suffixed. Losses named in ``uni_losses`` use the union matches and count in
        ``shared`` instead of ``indices`` / ``num_boxes``.
        """
        result = {}
        for loss in self.losses if losses is None else losses:
            ind, nb = shared if (self.use_uni_set and loss in uni_losses) else (indices, num_boxes)
            meta = self.get_loss_meta_info(loss, outputs, targets, ind)
            l_dict = self.get_loss(loss, outputs, targets, ind, nb, batch_queries_num=batch_queries_num, **meta)
            result.update({k + suffix: v * self.weight_dict[k] for k, v in l_dict.items() if k in self.weight_dict})
        return result

    @staticmethod
    def _average_over_ranks(count, device) -> float:
        """A count averaged over the distributed ranks, at least 1."""
        count = torch.as_tensor([count], dtype=torch.float, device=device)
        if dist_utils.is_dist_available_and_initialized():
            torch.distributed.all_reduce(count)
        return torch.clamp(count / dist_utils.get_world_size(), min=1).item()

    @staticmethod
    def _union_indices(indices, indices_aux_list):
        """
        D-FINE's 'go' matching: the union of one set of matches with several others, keeping for
        every query the target it was matched to most often.
        """
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices, indices_aux)
            ]
        results = []
        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            unique_sorted = unique[torch.argsort(counts, descending=True)]
            query_to_target = {}
            for row_idx, col_idx in unique_sorted.tolist():
                query_to_target.setdefault(row_idx, col_idx)
            rows = torch.tensor(list(query_to_target.keys()), device=ind.device)
            cols = torch.tensor(list(query_to_target.values()), device=ind.device)
            results.append((rows.long(), cols.long()))
        return results

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
    def _class_agnostic(self, targets):
        """Score against a single class: every label becomes 0 and ``num_classes`` is 1 for the duration."""
        num_classes = self.num_classes
        self.num_classes = 1
        agnostic_targets = copy.deepcopy(targets)
        for t in agnostic_targets:
            t["labels"] = torch.zeros_like(t["labels"])
        try:
            yield agnostic_targets
        finally:
            self.num_classes = num_classes

    def forward(self, outputs, targets, **kwargs):
        assert "aux_outputs" in outputs, "DomeCriterion needs the decoder's auxiliary outputs (aux_loss: True)"
        device = outputs["pred_logits"].device
        batch_queries_num = outputs.get("batch_queries_num")
        self._clear_cache()

        # match every prediction set, and build the union matching for the box losses
        main_outputs = {k: v for k, v in outputs.items() if "aux" not in k}
        match = lambda o: self.matcher(o, targets, batch_queries_num=batch_queries_num)["indices"]  # noqa: E731
        indices = match(main_outputs)
        cached_indices = [match(o) for o in outputs["aux_outputs"] + [outputs["pre_outputs"]]]
        cached_indices_enc = [match(o) for o in outputs["enc_aux_outputs"]]
        indices_go = self._union_indices(indices, cached_indices + cached_indices_enc)
        shared = (indices_go, self._average_over_ranks(sum(len(x[0]) for x in indices_go), device))
        num_boxes = self._average_over_ranks(sum(len(t["labels"]) for t in targets), device)

        def block(set_outputs, set_targets, set_indices, suffix, uni_losses=("boxes", "local"), **overrides):
            args = dict(num_boxes=num_boxes, shared=shared, batch_queries_num=batch_queries_num)
            args.update(overrides)
            return self._weighted_losses(
                set_outputs, set_targets, set_indices, suffix=suffix, uni_losses=uni_losses, **args
            )

        losses = block(outputs, targets, indices, "")

        for i, aux in enumerate(outputs["aux_outputs"]):
            if "local" in self.losses:
                aux["up"], aux["reg_scale"] = outputs["up"], outputs["reg_scale"]
                aux["fdr_min_unit"] = outputs.get("fdr_min_unit")
            losses.update(block(aux, targets, cached_indices[i], f"_aux_{i}"))

        losses.update(block(outputs["pre_outputs"], targets, cached_indices[-1], "_pre"))

        enc_args = dict(uni_losses=("boxes",), losses=self.enc_losses)
        if outputs["enc_meta"]["class_agnostic"]:
            with self._class_agnostic(targets) as enc_targets:
                for i, enc in enumerate(outputs["enc_aux_outputs"]):
                    losses.update(block(enc, enc_targets, cached_indices_enc[i], f"_enc_{i}", **enc_args))
        else:
            for i, enc in enumerate(outputs["enc_aux_outputs"]):
                losses.update(block(enc, targets, cached_indices_enc[i], f"_enc_{i}", **enc_args))

        if "dn_outputs" in outputs:
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]
            dn_args = dict(uni_losses=(), num_boxes=dn_num_boxes, batch_queries_num=None)
            for i, dn in enumerate(outputs["dn_outputs"]):
                if "local" in self.losses:
                    dn["is_dn"] = True
                    dn["up"], dn["reg_scale"] = outputs["up"], outputs["reg_scale"]
                    dn["fdr_min_unit"] = outputs.get("fdr_min_unit")
                losses.update(block(dn, targets, indices_dn, f"_dn_{i}", **dn_args))
            losses.update(block(outputs["dn_pre_outputs"], targets, indices_dn, "_dn_pre", **dn_args))

        if "defe" in outputs:
            losses.update(self.loss_defe(outputs["defe"], targets))

        # a NaN term must not take the whole step down with it
        return {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
