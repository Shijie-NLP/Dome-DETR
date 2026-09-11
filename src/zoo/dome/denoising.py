"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch

from ...misc.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from ...nn.functional import inverse_sigmoid

__all__ = ["get_contrastive_denoising_training_group"]


def get_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    batch_queries_num=None,
):
    """
    Contrastive denoising queries (DINO / RT-DETR): the ground-truth boxes of every image,
    padded to the largest count in the batch and repeated in ``num_group`` groups of a positive
    (lightly jittered) and a negative (heavily jittered) copy, with a share of the labels swapped
    at random. Returns their class embeddings, their boxes as logits, the self-attention mask, and
    a ``dn_meta`` dict (``dn_positive_idx``, ``dn_num_group``, ``dn_num_split``).

    The attention mask is ``[bs, 1, T, T]``, True where attention is blocked (the same for
    every head): the matching queries cannot see the denoising queries, denoising groups cannot see each other,
    and, with ``batch_queries_num`` (the real query count of every image, the rest being
    padding), padding queries and real queries are hidden from one another. Padding queries still
    see themselves so their softmax stays finite.
    """
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t["labels"]) for t in targets]
    device = targets[0]["labels"].device
    bs = len(num_gts)
    max_gt_num = max(num_gts)

    if max_gt_num == 0:
        # no boxes in the whole batch: empty tensors, but still run class_embed so it stays in the graph
        input_query_class = torch.full([bs, 0], num_classes, dtype=torch.int32, device=device)
        input_query_logits = class_embed(input_query_class)
        input_query_bbox_unact = inverse_sigmoid(torch.zeros([bs, 0, 4], device=device))
        attn_mask = torch.zeros([bs, 1, num_queries, num_queries], dtype=torch.bool, device=device)
        dn_meta = {"dn_positive_idx": None, "dn_num_group": 0, "dn_num_split": [0, num_queries]}
        return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta

    num_group = max(num_denoising // max_gt_num, 1)

    # the ground truths padded to the largest count (no syncs: the counts are host values)
    pad = torch.nn.utils.rnn.pad_sequence
    input_query_class = pad([t["labels"] for t in targets], batch_first=True, padding_value=num_classes).to(torch.int32)
    input_query_bbox = pad([t["boxes"] for t in targets], batch_first=True)
    pad_gt_mask = (torch.arange(max_gt_num)[None, :] < torch.tensor(num_gts)[:, None]).to(device, non_blocking=True)

    # each group has positive and negative queries
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    # the positive query of every ground truth in every group, per image: group after group,
    # ground truths in order (what nonzero over the positive mask listed)
    positions = torch.arange(num_group, device=device)[:, None] * (2 * max_gt_num) + torch.arange(
        max_gt_num, device=device
    )
    dn_positive_idx = tuple(positions[:, :n].reshape(-1) for n in num_gts)
    num_denoising = int(max_gt_num * 2 * num_group)  # total denoising queries

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        # negatives are pushed at least a full step away, positives less than one
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        known_bbox += rand_sign * rand_part * diff
        known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
        # negative sizes flipped positive (upstream's ``x[x < 0] *= -1``, whose boolean index
        # syncs the host mid-forward, on the encoder)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox).abs()
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)

    input_query_logits = class_embed(input_query_class)

    tgt_size = num_denoising + num_queries
    # a denoising query is seen only by the queries of its own group: not by the matching
    # queries, not by the other groups
    position = torch.arange(tgt_size, device=device)
    group = torch.where(position < num_denoising, position // (2 * max_gt_num), -1)
    attn_mask = (position[None, :] < num_denoising) & (group[:, None] != group[None, :])
    attn_mask = attn_mask[None, None].expand(bs, 1, -1, -1)  # [bs, 1, T, T]

    if batch_queries_num is not None:
        # real queries and padding do not see one another (padding still sees itself)
        counts = torch.tensor(batch_queries_num).to(device, non_blocking=True)
        real = position[None, :] < num_denoising + counts[:, None]  # [bs, T]
        attn_mask = attn_mask | (real[:, None, :, None] != real[:, None, None, :])

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries],
    }
    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta
