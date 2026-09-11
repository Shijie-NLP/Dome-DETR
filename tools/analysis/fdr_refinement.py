"""
How the decoder refines its boxes layer by layer (D-FINE's Fine-grained Distribution
Refinement, FDR). A trained model is run on validation images with every decoder layer's
output captured, and each ground-truth object is followed through the stages of its detection,
the highest-scoring query whose final box reaches the IoU threshold with it: the encoder proposal, the first layer's plain box (the anchor of every distribution),
then the FDR box of each layer, whose four edges are expectations of a distribution over the
``reg_max + 1`` bins of W(n).

Written into ``<run>/fdr/``:

* ``object_<k>.png``: one object of the chosen image, the boxes of every stage on a crop and
  the four edge distributions of every layer (expectation and the ground-truth edge marked).
* ``image.png``: the image with ground truth, final detections and the objects that got a figure.
* ``trend.png`` and ``trend.md``: over a sample of images, per object size, how IoU with the
  ground truth, edge movement, distribution sharpness and score move from stage to stage; then
  the geometry of the edge range: how far the ground-truth edges are from the anchor's against
  how far the distributions can reach, the finest bin against the last layer's edge error, and
  what a floor of the edge unit (``min_refine_cells``) would change, without retraining.

    python tools/analysis/fdr_refinement.py outputs/dfine_s_visdrone/2026-09-09_17-13-58
    python tools/analysis/fdr_refinement.py outputs/dfine_s_visdrone/2026-09-09_17-13-58 --image 12 --objects 8 --num-images 100
    python tools/analysis/fdr_refinement.py <run> --checkpoint best_stg1.pth --image 0000001_02999_d_0000005
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.core import YAMLConfig  # noqa: E402
from src.misc.box_ops import box_cxcywh_to_xyxy, box_iou  # noqa: E402
from src.zoo.dome.dfine_decoder import TransformerDecoder  # noqa: E402
from src.zoo.dome.fdr import _edge_unit, weighting_function  # noqa: E402

SIZE_BUCKETS = [("tiny <16px", 0, 16), ("small 16-32px", 16, 32), ("medium 32-96px", 32, 96), ("large >96px", 96, 1e9)]
EDGES = ["left", "top", "right", "bottom"]
# stage colours: proposal and anchor in grey/orange, the FDR layers in a blue ramp, ground truth black
PROPOSAL, ANCHOR, GT = "#898781", "#eb6834", "#0b0b0b"
LAYER_RAMP = ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e6e5e1"


# --------------------------------------------------------------------------------------------
# model and data
# --------------------------------------------------------------------------------------------


def pick_checkpoint(run_dir, name):
    if name:
        return name if os.path.isabs(name) or os.path.exists(name) else os.path.join(run_dir, name)
    for candidate in ("best_stg2.pth", "best_stg1.pth", "last.pth"):
        path = os.path.join(run_dir, candidate)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"no checkpoint in {run_dir}")


def load_model(cfg, checkpoint, device):
    """The EMA weights of ``checkpoint`` (the ones validation scores), else the raw model's."""
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state["ema"]["module"] if "ema" in state else state["model"]
    model = cfg.model
    model.load_state_dict(weights)
    return model.to(device).eval()


def find_decoder(model):
    decoders = [m for m in model.modules() if isinstance(m, TransformerDecoder)]
    if len(decoders) != 1:
        raise RuntimeError(f"expected one TransformerDecoder in the model, found {len(decoders)}")
    return decoders[0]


@torch.no_grad()
def run_all_layers(model, decoder, images):
    """
    Every decoder layer's boxes, logits and edge distributions for ``images``. Evaluation stops
    at ``eval_idx``, so the inner decoder alone is switched to its training branch (its layers
    keep evaluation behaviour: dropout stays off, no denoising queries are added).

    Returns a dict of tensors on the CPU, batch first: ``proposal`` [B,Q,4] the encoder's
    proposals, ``anchor`` [B,Q,4] the first layer's plain boxes, ``boxes`` [B,L,Q,4] the FDR
    boxes of each layer, ``corners`` [B,L,Q,4,reg_max+1] their cumulative bin logits, ``logits``
    [B,L,Q,C] and ``pre_logits`` [B,Q,C] the class scores, ``min_unit`` the floor of the edge unit.
    """
    captured = {}

    def hook(module, args, kwargs, output):
        captured["proposal"] = torch.sigmoid(args[1])
        captured["min_unit"] = kwargs.get("fdr_min_unit")
        captured["output"] = output

    handle = decoder.register_forward_hook(hook, with_kwargs=True)
    decoder.training = True  # this module's flag only, not its children's
    try:
        model(images)
    finally:
        decoder.training = False
        handle.remove()

    boxes, logits, corners, refs, pre_boxes, pre_logits = captured["output"]
    reg_max = decoder.reg_max
    return {
        "proposal": captured["proposal"].float().cpu(),
        "anchor": pre_boxes.float().cpu(),
        "boxes": boxes.transpose(0, 1).float().cpu(),
        "logits": logits.transpose(0, 1).float().cpu(),
        "corners": corners.transpose(0, 1).float().cpu().unflatten(-1, (4, reg_max + 1)),
        "pre_logits": pre_logits.float().cpu(),
        "min_unit": None if captured["min_unit"] is None else captured["min_unit"].float().cpu(),
    }


def stage_boxes(out, b):
    """The boxes of image ``b`` at every stage, [S, Q, 4] cxcywh normalized, and the stage names."""
    boxes = torch.cat([out["proposal"][b][None], out["anchor"][b][None], out["boxes"][b]])
    names = ["proposal", "anchor"] + [f"layer {i}" for i in range(out["boxes"].shape[1])]
    return boxes, names


def stage_scores(out, b):
    """The best class probability of every query at the anchor and each layer, [L+1, Q]."""
    logits = torch.cat([out["pre_logits"][b][None], out["logits"][b]])
    return logits.sigmoid().max(-1).values


def gt_distances(anchor, gt_xyxy, reg_scale, min_unit):
    """The ground-truth edges as FDR distances from the anchor box: [N, 4], the unit ``bbox2distance`` uses."""
    unit = _edge_unit(anchor, reg_scale, min_unit)
    left = (anchor[:, 0] - 0.5 * anchor[:, 2] - gt_xyxy[:, 0]) / unit[:, 0]
    top = (anchor[:, 1] - 0.5 * anchor[:, 3] - gt_xyxy[:, 1]) / unit[:, 1]
    right = (gt_xyxy[:, 2] - anchor[:, 0] - 0.5 * anchor[:, 2]) / unit[:, 0]
    bottom = (gt_xyxy[:, 3] - anchor[:, 1] - 0.5 * anchor[:, 3]) / unit[:, 1]
    return torch.stack([left, top, right, bottom], -1), unit


FLOORS = [0, 1, 2, 4]  # min_refine_cells values the what-if table evaluates


def range_stats(sample, q, g, floor_px=0.0):
    """
    The edge range of query ``q``'s anchor against ground truth ``g``, in input px: the shift each
    edge needs, the furthest the distributions reach (``W(reg_max)`` units), the finest bin, and
    the last layer's edge error. ``floor_px`` floors the anchor size in the unit as
    ``min_refine_cells`` would (0: the run's own floor, if any).
    """
    anchor = sample.boxes_norm[1, q][None]
    gt_norm = sample.gt[g][None] / sample.size.repeat(2)
    min_unit = sample.out["min_unit"]
    if floor_px > 0:
        min_unit = torch.tensor([floor_px, floor_px]) / sample.size
    needed, unit = gt_distances(anchor, gt_norm, sample.reg_scale, min_unit)
    needed, unit_px = needed[0], (unit[0] * sample.size).repeat(2)  # [4] W units, [4] px per unit (x, y, x, y)
    weights = sample.weights
    max_w, mid = float(weights[-1]), len(weights) // 2
    finest_px = float(weights[mid + 1] - weights[mid]) * unit_px
    beyond = needed.abs() > max_w
    expectation = (sample.probs[-1, q] * torch.from_numpy(weights)).sum(-1)  # [4]
    err_px = (expectation - needed).abs() * unit_px
    return {
        "anchor_iou": float(box_iou(sample.boxes[1, q][None], sample.gt[g][None])[0][0, 0]),
        "needed_px": float((needed.abs() * unit_px).mean()),
        "reach_px": float((max_w * unit_px).mean()),
        "ratio": float((needed.abs() / max_w).max()),
        "beyond": float(beyond.float().mean()),
        "any_beyond": float(beyond.any()),
        "finest_px": float(finest_px.mean()),
        "err_px": float(err_px.mean()),
        "err_below_bin": float((err_px < finest_px).float().mean()),
    }


def range_records_of(sample, iou_threshold, finest_stride):
    """
    One record per ground-truth box: its size bucket, the range statistics through its
    detection's query (``det``, None when it is not detected) and, for every floor of ``FLOORS``
    in cells of ``finest_stride``, through the query whose anchor overlaps it most (``floors``).
    """
    if len(sample.gt) == 0:
        return []
    detected = {g: q for g, q, _ in sample.match(iou_threshold)}
    iou, _ = box_iou(sample.gt, sample.boxes[1])  # [G, Q] against the anchors
    best_iou, best_q = iou.max(1)
    gt_centre = (sample.gt[:, :2] + sample.gt[:, 2:]) / 2
    anchor_centre = (sample.boxes[1][:, :2] + sample.boxes[1][:, 2:]) / 2
    nearest = torch.cdist(gt_centre, anchor_centre).argmin(1)
    records = []
    for g in range(len(sample.gt)):
        q = int(best_q[g]) if best_iou[g] > 0 else int(nearest[g])
        records.append(
            {
                "bucket": bucket_of(sample.gt_size(g)),
                "det": range_stats(sample, detected[g], g) if g in detected else None,
                "floors": {k: range_stats(sample, q, g, k * finest_stride) for k in FLOORS},
            }
        )
    return records


def range_table(records, finest_stride):
    lines = []
    names = [b[0] for b in SIZE_BUCKETS] + ["all"]

    def rows_of(name):
        return records if name == "all" else [r for r in records if r["bucket"] == name]

    for title, pick in (
        ("Edge reach of the anchor, detected objects (through their detection's query)", lambda r: r["det"]),
        (
            "Edge reach of the anchor, all ground truth (through the query whose anchor overlaps most)",
            lambda r: r["floors"][0],
        ),
    ):
        lines.append(f"### {title}\n")
        lines.append(
            "| size | n | anchor IoU | needed shift px | reach px | needed/reach p50 | p90 | edges beyond reach | objects with an edge beyond reach |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for name in names:
            st = [pick(r) for r in rows_of(name) if pick(r) is not None]
            if not st:
                continue
            ratios = np.array([x["ratio"] for x in st])
            lines.append(
                f"| {name} | {len(st)} | {np.mean([x['anchor_iou'] for x in st]):.2f} | {np.mean([x['needed_px'] for x in st]):.1f} "
                f"| {np.mean([x['reach_px'] for x in st]):.1f} | {np.median(ratios):.2f} | {np.percentile(ratios, 90):.2f} "
                f"| {np.mean([x['beyond'] for x in st]):.1%} | {np.mean([x['any_beyond'] for x in st]):.1%} |"
            )
        lines.append("")

    lines.append("### Resolution at the last layer, detected objects\n")
    lines.append("| size | n | finest bin px | final edge error px | edges with an error below the finest bin |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name in names:
        st = [r["det"] for r in rows_of(name) if r["det"] is not None]
        if not st:
            continue
        lines.append(
            f"| {name} | {len(st)} | {np.mean([x['finest_px'] for x in st]):.2f} | {np.mean([x['err_px'] for x in st]):.2f} "
            f"| {np.mean([x['err_below_bin'] for x in st]):.0%} |"
        )
    lines.append("")

    lines.append(
        f"### A floor of the edge unit at k cells of stride {finest_stride} (min_refine_cells), all ground truth\n"
    )
    lines.append("| size | n | k | reach px | finest bin px | edges beyond reach | objects with an edge beyond reach |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in names:
        rows = rows_of(name)
        if not rows:
            continue
        for k in FLOORS:
            st = [r["floors"][k] for r in rows]
            lines.append(
                f"| {name} | {len(rows)} | {k} | {np.mean([x['reach_px'] for x in st]):.1f} | {np.mean([x['finest_px'] for x in st]):.2f} "
                f"| {np.mean([x['beyond'] for x in st]):.1%} | {np.mean([x['any_beyond'] for x in st]):.1%} |"
            )
    lines.append("")
    return "\n".join(lines)


def value_to_bin(values, weights):
    """Fractional bin positions of ``values`` on the monotone W(n) ``weights``, for drawing."""
    return np.interp(np.asarray(values), np.asarray(weights), np.arange(len(weights)))


class Sample:
    """One validation image through the model, with everything the figures and statistics need."""

    def __init__(self, dataset, collate, index, model, decoder, device):
        image, target = dataset[index]
        images = collate([(image, target)])[0]
        self.index = index
        self.name = (
            dataset.file_name(dataset.hf_meta[index]["id"]) if "id" in dataset.hf_meta.column_names else str(index)
        )
        self.image = images[0].permute(1, 2, 0).numpy()  # HWC float, the padded network input
        self.size = torch.tensor([images.shape[-1], images.shape[-2]], dtype=torch.float32)  # (w, h) of the input
        self.gt = target["boxes"].as_subclass(torch.Tensor).float()  # xyxy in input pixels
        self.gt_labels = target["labels"]
        self.orig_scale = (
            (target["orig_size"].float() / target.get("resized_size", target["orig_size"]).float()).mean().item()
        )

        self.out = run_all_layers(model, decoder, images.to(device))
        self.boxes_norm, self.stages = stage_boxes(self.out, 0)  # [S, Q, 4]
        scale = self.size.repeat(2)
        self.boxes = box_cxcywh_to_xyxy(self.boxes_norm) * scale  # [S, Q, 4] xyxy input pixels
        self.scores = stage_scores(self.out, 0)  # [L+1, Q]
        self.reg_scale = abs(decoder.reg_scale.detach().float().cpu())
        self.weights = weighting_function(decoder.reg_max, decoder.up.detach().float().cpu(), self.reg_scale).numpy()
        self.probs = self.out["corners"][0].softmax(-1)  # [L, Q, 4, bins]

    def match(self, iou_threshold):
        """
        Per ground-truth box its detection: of the queries whose final box reaches ``iou_threshold``
        with it, the one with the highest score, as (gt index, query, IoU). The best-overlapping
        query is often a suppressed duplicate (one-to-one matching trains it as background), so
        the score picks the query the postprocessor would keep. Boxes without such a query are left out.
        """
        if len(self.gt) == 0:
            return []
        iou, _ = box_iou(self.gt, self.boxes[-1])  # [G, Q]
        scores = self.scores[-1][None].expand_as(iou)
        best, query = torch.where(iou >= iou_threshold, scores, -1.0).max(1)
        return [(g, int(query[g]), float(iou[g, query[g]])) for g in range(len(self.gt)) if best[g] >= 0]

    def gt_size(self, g):
        """Square-root area of ground-truth box ``g`` in original-image pixels."""
        w, h = self.gt[g, 2] - self.gt[g, 0], self.gt[g, 3] - self.gt[g, 1]
        return float((w * h).sqrt() * self.orig_scale)

    def query_distances(self, q, g):
        """FDR distances of query ``q``: the expectation of every layer [L, 4], the ground-truth edges [4], the unit in input px [2]."""
        anchor = self.boxes_norm[1, q][None]
        gt_norm = self.gt[g][None] / self.size.repeat(2)
        gt, unit = gt_distances(anchor, gt_norm, self.reg_scale, self.out["min_unit"])
        expectation = (self.probs[:, q] * torch.from_numpy(self.weights)).sum(-1)  # [L, 4]
        return expectation, gt[0], unit[0] * self.size


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def stage_color(s, num_layers):
    if s == 0:
        return PROPOSAL
    if s == 1:
        return ANCHOR
    ramp = LAYER_RAMP[-num_layers:] if num_layers <= len(LAYER_RAMP) else LAYER_RAMP
    return ramp[min(s - 2, len(ramp) - 1)]


def draw_box(ax, box, color, linestyle="-", linewidth=1.4, label=None):
    from matplotlib.patches import Rectangle

    x1, y1, x2, y2 = [float(v) for v in box]
    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
    )


def object_figure(sample, g, q, path, names):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_layers = sample.probs.shape[0]
    boxes = sample.boxes[:, q]  # [S, 4]
    gt = sample.gt[g]
    expectation, gt_dist, unit = sample.query_distances(q, g)
    bins = np.arange(len(sample.weights))

    fig = plt.figure(figsize=(4.2 + 2.6 * 4, 1.0 + 1.7 * num_layers), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    grid = fig.add_gridspec(num_layers, 5, width_ratios=[1.6, 1, 1, 1, 1])

    # the crop: every stage's box around the ground truth
    ax = fig.add_subplot(grid[:, 0])
    all_boxes = torch.cat([boxes, gt[None]])
    x1, y1 = all_boxes[:, :2].min(0).values
    x2, y2 = all_boxes[:, 2:].max(0).values
    margin = max(float(x2 - x1), float(y2 - y1)) * 0.5 + 8
    h, w = sample.image.shape[:2]
    cx1, cy1 = max(0, int(x1 - margin)), max(0, int(y1 - margin))
    cx2, cy2 = min(w, int(x2 + margin) + 1), min(h, int(y2 + margin) + 1)
    ax.imshow(np.clip(sample.image[cy1:cy2, cx1:cx2], 0, 1), extent=(cx1, cx2, cy2, cy1), interpolation="nearest")
    for s in range(len(boxes)):
        draw_box(
            ax, boxes[s], stage_color(s, num_layers), "--" if s < 2 else "-", 1.2 if s < 2 else 1.4, sample.stages[s]
        )
    draw_box(ax, gt, GT, ":", 1.6, "ground truth")
    ax.set_xlim(cx1, cx2)
    ax.set_ylim(cy2, cy1)
    ax.set_xticks([])
    ax.set_yticks([])
    ious = [float(box_iou(b[None], gt[None])[0][0, 0]) for b in boxes]
    label = names.get(int(sample.gt_labels[g]), str(int(sample.gt_labels[g])))
    ax.set_title(
        f"{label}, {sample.gt_size(g):.0f}px; IoU "
        + " > ".join(f"{iou:.2f}" for iou in ious)
        + f"; score {sample.scores[-1, q]:.2f}",
        fontsize=8,
        color=INK,
        loc="left",
    )
    ax.legend(fontsize=6, frameon=False, loc="lower left", labelcolor=INK)

    # the distributions: one row per layer, one column per edge
    ticks = [0, len(bins) // 4, len(bins) // 2, 3 * len(bins) // 4, len(bins) - 1]
    top = float(sample.probs[:, q].max()) * 1.15  # one scale for every cell, so sharpening shows
    for layer in range(num_layers):
        color = stage_color(layer + 2, num_layers)
        for e in range(4):
            ax = fig.add_subplot(grid[layer, e + 1])
            style(ax)
            prob = sample.probs[layer, q, e].numpy()
            ax.bar(bins, prob, width=0.8, color=color, linewidth=0)
            ax.axvline(value_to_bin(expectation[layer, e], sample.weights), color=INK, linewidth=1.0)
            ax.axvline(value_to_bin(gt_dist[e], sample.weights), color=GT, linewidth=1.0, linestyle=":")
            px = unit[e % 2]
            ax.set_title(
                f"{EDGES[e]}: {float(expectation[layer, e] * px):+.1f}px (gt {float(gt_dist[e] * px):+.1f}px, top {prob.max():.2f})",
                fontsize=7,
                color=INK,
                loc="left",
            )
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{sample.weights[t]:.1f}" for t in ticks] if layer == num_layers - 1 else [])
            ax.set_ylim(0, top)
            if e == 0:
                ax.set_ylabel(f"layer {layer}", fontsize=8, color=INK)
            if layer == num_layers - 1:
                ax.set_xlabel("W(n), in units of anchor size / reg_scale", fontsize=7, color=MUTED)
    fig.suptitle(
        f"{sample.name}: query {q}. Solid line = expectation of the distribution, dotted = ground-truth edge; outwards positive.",
        fontsize=8,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def image_figure(sample, chosen, score_threshold, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = sample.image.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(np.clip(sample.image, 0, 1))
    for box in sample.gt:
        draw_box(ax, box, GT, ":", 0.8)
    keep = sample.scores[-1] >= score_threshold
    for box in sample.boxes[-1][keep]:
        draw_box(ax, box, LAYER_RAMP[-1], "-", 0.8)
    for k, (g, _, _) in enumerate(chosen):
        box = sample.gt[g]
        draw_box(ax, box, ANCHOR, "-", 1.6)
        ax.text(float(box[0]), float(box[1]) - 2, str(k), fontsize=9, color=ANCHOR, fontweight="bold")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.set_title(
        f"{sample.name}: ground truth dotted, detections with score >= {score_threshold} solid, numbered objects have a figure",
        fontsize=9,
        loc="left",
    )
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def trend_figure(records, stages, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buckets = [b for b in SIZE_BUCKETS if any(r["bucket"] == b[0] for r in records)]
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    panels = [
        ("IoU with the ground truth", "iou", stages, "IoU"),
        ("edge movement from the previous stage", "move", stages[1:], "mean |edge shift|, input px"),
        ("distribution sharpness", "top1", stages[2:], "mean top-1 bin probability"),
        ("score", "score", stages[1:], "best class probability"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (title, key, xs, ylabel) in zip(axes, panels):
        style(ax)
        for i, (name, _, _) in enumerate(buckets):
            rows = [r for r in records if r["bucket"] == name]
            values = np.array([r[key] for r in rows]).mean(0)
            ax.plot(
                range(len(xs)),
                values,
                color=palette[i],
                linewidth=1.6,
                marker="o",
                markersize=4,
                label=f"{name} (n={len(rows)})",
            )
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, fontsize=8)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
        if key == "move":
            ax.set_yscale("log")
        ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def trend_table(records, stages):
    lines = []
    for key, title, xs, fmt in (
        ("iou", "Mean IoU with the ground truth", stages, "{:.3f}"),
        ("move", "Mean edge shift from the previous stage, input pixels", stages[1:], "{:.2f}"),
        ("top1", "Mean top-1 bin probability of the edge distributions", stages[2:], "{:.3f}"),
        ("score", "Mean best class probability", stages[1:], "{:.3f}"),
    ):
        lines.append(f"### {title}\n")
        lines.append("| size | n | " + " | ".join(xs) + " |")
        lines.append("| --- | ---: | " + " | ".join("---:" for _ in xs) + " |")
        for name, _, _ in SIZE_BUCKETS + [("all", 0, 1e9)]:
            rows = records if name == "all" else [r for r in records if r["bucket"] == name]
            if not rows:
                continue
            values = np.array([r[key] for r in rows]).mean(0)
            lines.append(f"| {name} | {len(rows)} | " + " | ".join(fmt.format(v) for v in values) + " |")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------


def bucket_of(size):
    return next(name for name, lo, hi in SIZE_BUCKETS if lo <= size < hi)


def records_of(sample, iou_threshold):
    """One record per matched ground-truth object: its size bucket and the per-stage series."""
    records = []
    num_layers = sample.probs.shape[0]
    for g, q, _ in sample.match(iou_threshold):
        boxes = sample.boxes[:, q]  # [S, 4]
        ious = [float(box_iou(b[None], sample.gt[g][None])[0][0, 0]) for b in boxes]
        moves = [float((boxes[s] - boxes[s - 1]).abs().mean()) for s in range(1, len(boxes))]
        top1 = [float(sample.probs[layer, q].max(-1).values.mean()) for layer in range(num_layers)]
        records.append(
            {
                "bucket": bucket_of(sample.gt_size(g)),
                "iou": ious,
                "move": moves,
                "top1": top1,
                "score": sample.scores[:, q].tolist(),
            }
        )
    return records


def resolve_image(dataset, image):
    if image is None:
        return 0
    if image.isdigit():
        return int(image)
    if "id" in dataset.hf_meta.column_names:
        ids = dataset.hf_meta["id"]
        hits = [i for i, row_id in enumerate(ids) if image in row_id]
        if len(hits) == 1:
            return hits[0]
        raise ValueError(f"{len(hits)} images match {image!r}")
    raise ValueError("this dataset has no image ids; pass a row index")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="output directory of train.py (config.yml and checkpoints)")
    parser.add_argument("--checkpoint", help="checkpoint file in the run (default: best_stg2, best_stg1, then last)")
    parser.add_argument(
        "--image",
        help="validation image for the object figures: a row index or a unique part of the image id (default: row 0)",
    )
    parser.add_argument(
        "--objects", type=int, default=6, help="objects of that image to draw, spread over sizes (default 6)"
    )
    parser.add_argument(
        "--num-images", type=int, default=50, help="images sampled for the trend statistics (default 50, 0 to skip)"
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="final IoU a query needs to be a candidate for an object; the highest-scoring one is taken (default 0.5)",
    )
    parser.add_argument(
        "--score", type=float, default=0.3, help="score threshold of the detections drawn on image.png (default 0.3)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", help="output directory (default <run>/fdr)")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(args.run, "fdr")
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = pick_checkpoint(args.run, args.checkpoint)
    print(f"checkpoint {checkpoint}")

    cfg = YAMLConfig(os.path.join(args.run, "config.yml"))
    model = load_model(cfg, checkpoint, args.device)
    decoder = find_decoder(model)
    loader = cfg.val_dataloader
    dataset, collate = loader.dataset, loader.collate_fn
    names = dict(getattr(dataset, "CATEGORIES", []))
    stages = ["proposal", "anchor"] + [f"layer {i}" for i in range(decoder.num_layers)]
    finest_stride = int(model.decoder.feat_strides[0])
    print(
        f"decoder: {decoder.num_layers} layers, reg_max {decoder.reg_max}, eval_idx {decoder.eval_idx}; {len(dataset)} validation images"
    )

    # the object figures
    index = resolve_image(dataset, args.image)
    sample = Sample(dataset, collate, index, model, decoder, args.device)
    matched = sorted(sample.match(args.match_iou), key=lambda m: sample.gt_size(m[0]))
    if not matched:
        print(f"{sample.name}: no ground-truth box has a query at IoU >= {args.match_iou}")
    picks = (
        np.unique(np.linspace(0, len(matched) - 1, min(args.objects, len(matched))).round().astype(int))
        if matched
        else []
    )
    chosen = [matched[i] for i in picks]
    for k, (g, q, iou) in enumerate(chosen):
        object_figure(sample, g, q, os.path.join(out_dir, f"object_{k}.png"), names)
        print(
            f"  object_{k}.png: gt {g} ({names.get(int(sample.gt_labels[g]), '?')}, {sample.gt_size(g):.0f}px), query {q}, final IoU {iou:.2f}"
        )
    image_figure(sample, chosen, args.score, os.path.join(out_dir, "image.png"))

    # the trend over a sample of images
    if args.num_images > 0:
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(dataset)), min(args.num_images, len(dataset)))
        records, range_records = [], []
        for i, idx in enumerate(indices):
            s = sample if idx == index else Sample(dataset, collate, idx, model, decoder, args.device)
            records += records_of(s, args.match_iou)
            range_records += range_records_of(s, args.match_iou, finest_stride)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(indices)} images, {len(records)} objects")
        if records:
            trend_figure(records, stages, os.path.join(out_dir, "trend.png"))
            text = (
                f"# FDR refinement of `{os.path.relpath(checkpoint).replace(os.sep, '/')}`\n\n"
                f"{len(records)} ground-truth objects of {len(indices)} validation images, each followed through the query whose "
                f"final box reaches IoU {args.match_iou} with it and scores highest. Stages: the encoder proposal, the first layer's plain box "
                f"(the anchor of the distributions), then the FDR box of every layer. Sizes are square-root areas in original pixels.\n\n"
                f"![trend](trend.png)\n\n" + trend_table(records, stages) + "\n"
                f"The range tables are geometry: every edge distribution spans W(0) .. W({decoder.reg_max}) units of the anchor "
                f"size / reg_scale, so 'reach' is the furthest an edge can move from the anchor's, 'needed shift' how far the "
                f"ground-truth edge is, and 'beyond reach' a ground-truth edge no distribution can express. The finest bin is "
                f"W(1) - W(0) around zero in px; the expectation can land between bins. The floor table recomputes the "
                f"geometry with the unit floored as min_refine_cells would, on this model's anchors.\n\n"
                + range_table(range_records, finest_stride)
            )
            with open(os.path.join(out_dir, "trend.md"), "w", encoding="utf-8") as f:
                f.write(text)
            print(text)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
