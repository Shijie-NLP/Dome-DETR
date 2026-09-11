"""
Where the decoder's deformable cross-attention looks, layer by layer. A trained model is run
on validation images with every layer's sampling locations and attention weights captured,
and each ground-truth object is followed through its detection's query (the highest-scoring
query whose final box reaches the IoU threshold, as ``fdr_refinement.py`` does), so the
``object_<k>.png`` figures of the two tools show the same objects.

Written into ``<run>/attention/``:

* ``object_<k>.png``: one object, one panel per layer: the crop, the box the layer starts from,
  the ground truth, and every sampling point of the layer (all heads), coloured by feature
  level and sized by its attention weight.
* ``trend.png`` and ``trend.md``: over a sample of images, per object size and layer, the
  attention-weighted density of sampling points in ground-truth box units (where the model
  reads relative to the object), the share of weight per feature level, the share inside the
  box, and how far out the points reach; then per level the box the layer starts from in cells,
  how far apart each head's points of that level land in cells, and the weight carried by heads
  whose points all fall within one cell (the coarse-level collapse ``min_sample_cells`` floors).

    python tools/analysis/deformable_attention.py outputs/dfine_s_visdrone/2026-09-09_17-13-58
    python tools/analysis/deformable_attention.py <run> --image 0000001_02999_d_0000005 --objects 8 --num-images 100
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fdr_refinement import (  # noqa: E402
    ANCHOR,
    GRID,
    GT,
    INK,
    MUTED,
    SIZE_BUCKETS,
    SURFACE,
    Sample,
    bucket_of,
    draw_box,
    find_decoder,
    load_model,
    pick_checkpoint,
    resolve_image,
)

from src.core import YAMLConfig  # noqa: E402
from src.misc.box_ops import box_iou  # noqa: E402

LEVEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # one per feature level
REACH = 3.0  # the density plots span this many box sizes to either side of the centre


class SamplingRecorder:
    """
    Records what every decoder layer's cross-attention samples: per layer ``(locations
    [B, Q, H, P, 2] normalized to the input, weights [B, Q, H, P] softmaxed per head, level [P])``.
    Wraps the attention core, so the recorded locations are exactly the ones sampled, floors and
    all.
    """

    def __init__(self, decoder):
        self.decoder = decoder
        self.records = []
        self.originals = []

    def __enter__(self):
        for layer in self.decoder.layers:
            attn = layer.cross_attn
            original = attn.ms_deformable_attn_core

            def wrapped(value, shapes, locations, weights, num_points_list, _original=original, _attn=attn):
                self.records.append(
                    (
                        locations.detach().float().cpu(),
                        weights.detach().float().cpu(),
                        _attn.point_level.cpu(),
                        [tuple(s) for s in shapes],
                    )
                )
                return _original(value, shapes, locations, weights, num_points_list)

            attn.ms_deformable_attn_core = wrapped
            self.originals.append((attn, original))
        return self

    def __exit__(self, *exc):
        for attn, original in self.originals:
            attn.ms_deformable_attn_core = original


def sample_with_attention(dataset, collate, index, model, decoder, device):
    with SamplingRecorder(decoder) as recorder:
        sample = Sample(dataset, collate, index, model, decoder, device)
    if len(recorder.records) != decoder.num_layers:
        raise RuntimeError(f"recorded {len(recorder.records)} cross-attention calls for {decoder.num_layers} layers")
    sample.attention = recorder.records
    return sample


def query_points(sample, layer, q):
    """The layer's sampling points of query ``q``: positions [H*P, 2] in input px, weights [H*P] (sum H), levels [H*P]."""
    locations, weights, level, _ = sample.attention[layer]
    points = locations[0, q] * sample.size  # [H, P, 2]
    heads = points.shape[0]
    return points.reshape(-1, 2), weights[0, q].reshape(-1), level.repeat(heads)


def entering_box(sample, layer, q):
    """The reference box the layer's attention is centred on: the proposal for layer 0, else the previous layer's box."""
    return sample.boxes[0, q] if layer == 0 else sample.boxes[2 + layer - 1, q]


def object_stats(sample, layer, q, g):
    """Per level the share of attention weight, the share inside the ground truth, and the weighted reach in box units and px."""
    points, weights, levels = query_points(sample, layer, q)
    gt = sample.gt[g]
    total = float(weights.sum())
    center = (gt[:2] + gt[2:]) / 2
    size = gt[2:] - gt[:2]
    rel = (points - center) / size  # box units, centre 0, edges at +-0.5
    inside = (rel.abs() <= 0.5).all(-1)
    reach_units = rel.abs().max(-1).values  # Chebyshev distance in box units
    reach_px = (points - center).abs().max(-1).values
    num_levels = int(levels.max()) + 1
    return {
        "level_share": [float(weights[levels == lv].sum() / total) for lv in range(num_levels)],
        "inside": float(weights[inside].sum() / total),
        "within_1": float(weights[reach_units <= 1.0].sum() / total),
        "reach_units": float((weights * reach_units).sum() / total),
        "reach_px": float((weights * reach_px).sum() / total),
        "rel": rel,
        "weights": weights / total,
    }


def cell_stats(sample, layer, q, strides):
    """
    Per level: the half-size of the box the layer starts from in cells of that level, the
    attention-weighted spread (largest Chebyshev distance between a head's points of the level,
    in cells) and the share of the level's weight on heads whose points all lie within one cell.
    """
    locations, weights, level, _ = sample.attention[layer]
    points = locations[0, q] * sample.size  # [H, P, 2] input px
    w = weights[0, q]  # [H, P]
    box = entering_box(sample, layer, q)  # xyxy input px
    half_px = (box[2:] - box[:2]) / 2  # [2]
    half_cells, spread, collapsed = [], [], []
    for lv in range(int(level.max()) + 1):
        m = level == lv
        pts = points[:, m]  # [H, n, 2]
        wl = w[:, m].sum(-1)  # [H] the level's weight per head
        distance = (pts[:, :, None] - pts[:, None]).abs().max(-1).values  # [H, n, n] Chebyshev, px
        s = distance.flatten(1).max(-1).values / strides[lv]  # [H] cells
        total = float(wl.sum())
        half_cells.append(float(half_px.mean() / strides[lv]))
        spread.append(float((wl * s).sum() / total) if total > 0 else 0.0)
        collapsed.append(float(wl[s < 1.0].sum() / total) if total > 0 else 0.0)
    return {"half_cells": half_cells, "spread": spread, "collapsed": collapsed}


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------


def object_figure(sample, g, q, strides, path, names):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    num_layers = len(sample.attention)
    gt = sample.gt[g]
    size = float((gt[2:] - gt[:2]).max())
    margin = max(REACH * size / 2, 24.0)
    h, w = sample.image.shape[:2]
    cx, cy = float((gt[0] + gt[2]) / 2), float((gt[1] + gt[3]) / 2)
    x1, y1 = max(0, int(cx - margin)), max(0, int(cy - margin))
    x2, y2 = min(w, int(cx + margin) + 1), min(h, int(cy + margin) + 1)

    label = names.get(int(sample.gt_labels[g]), str(int(sample.gt_labels[g])))
    final_iou = float(box_iou(sample.boxes[-1, q][None], gt[None])[0][0, 0])
    header = (
        f"{sample.name}: query {q}, {label}, {sample.gt_size(g):.0f}px, final IoU {final_iou:.2f}, "
        f"score {sample.scores[-1, q]:.2f}. Marker area follows the attention weight; all heads shown."
    )
    fig, axes = plt.subplots(1, num_layers, figsize=(4.6 * num_layers, 5.0), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for layer, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(np.clip(sample.image[y1:y2, x1:x2], 0, 1), extent=(x1, x2, y2, y1), interpolation="nearest")
        points, weights, levels = query_points(sample, layer, q)
        for lv in range(int(levels.max()) + 1):
            m = levels == lv
            ax.scatter(
                points[m, 0],
                points[m, 1],
                s=4 + 300 * weights[m],  # weights sum to the head count over all points
                c=LEVEL_COLORS[lv % len(LEVEL_COLORS)],
                alpha=0.75,
                linewidths=0,
            )
        draw_box(ax, entering_box(sample, layer, q), ANCHOR, "--", 1.2)
        draw_box(ax, gt, GT, ":", 1.4)
        ax.set_xlim(x1, x2)
        ax.set_ylim(y2, y1)
        ax.set_xticks([])
        ax.set_yticks([])
        st = object_stats(sample, layer, q, g)
        shares = ", ".join(f"s{strides[lv]} {100 * s:.0f}%" for lv, s in enumerate(st["level_share"]))
        ax.set_title(
            (header if layer == 0 else "")
            + f"\nlayer {layer}: {shares}\ninside box {100 * st['inside']:.0f}%, within 1 box {100 * st['within_1']:.0f}%, reach {st['reach_units']:.2f} box ({st['reach_px']:.1f}px)",
            fontsize=8,
            color=INK,
            loc="left",
        )
    handles = [
        Line2D([], [], marker="o", linestyle="", color=LEVEL_COLORS[lv], label=f"stride {s}")
        for lv, s in enumerate(strides)
    ]
    handles += [
        Line2D([], [], color=ANCHOR, linestyle="--", label="box entering the layer"),
        Line2D([], [], color=GT, linestyle=":", label="ground truth"),
    ]
    np.atleast_1d(axes)[0].legend(handles=handles, fontsize=7, frameon=False, loc="lower left", labelcolor=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def trend_figure(records, num_layers, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    buckets = [b[0] for b in SIZE_BUCKETS if any(r["bucket"] == b[0] for r in records)]
    bins = np.linspace(-REACH, REACH, 61)
    fig, axes = plt.subplots(
        len(buckets), num_layers, figsize=(3.4 * num_layers, 3.4 * len(buckets)), constrained_layout=True, squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    for row, name in enumerate(buckets):
        rows = [r for r in records if r["bucket"] == name]
        for layer in range(num_layers):
            ax = axes[row, layer]
            rel = np.concatenate([r["rel"][layer] for r in rows])
            weights = np.concatenate([r["weights"][layer] for r in rows]) / len(rows)
            hist, _, _ = np.histogram2d(rel[:, 0], rel[:, 1], bins=[bins, bins], weights=weights)
            ax.imshow(hist.T, extent=(-REACH, REACH, REACH, -REACH), cmap="Blues", interpolation="nearest", vmin=0)
            ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor=GT, linestyle=":", linewidth=1.2))
            ax.set_xticks([-2, -1, 0, 1, 2])
            ax.set_yticks([-2, -1, 0, 1, 2])
            ax.tick_params(colors=MUTED, labelsize=7, length=0)
            for side in ax.spines.values():
                side.set_color(GRID)
            inside = np.mean([r["inside"][layer] for r in rows])
            ax.set_title(
                f"{name}, layer {layer}: {100 * inside:.0f}% inside (n={len(rows)})", fontsize=8, color=INK, loc="left"
            )
            if layer == 0:
                ax.set_ylabel("box heights from the centre", fontsize=7, color=MUTED)
            if row == len(buckets) - 1:
                ax.set_xlabel("box widths from the centre", fontsize=7, color=MUTED)
    fig.suptitle(
        "Attention-weighted density of sampling points in ground-truth box units (dotted: the box)",
        fontsize=9,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def trend_table(records, num_layers, strides):
    lines = []
    names = [b[0] for b in SIZE_BUCKETS] + ["all"]
    for key, title, fmt in (
        ("inside", "Share of attention weight inside the ground-truth box", "{:.0%}"),
        ("within_1", "Share within one box size of the centre (Chebyshev)", "{:.0%}"),
        ("reach_units", "Weighted mean reach, in box units", "{:.2f}"),
        ("reach_px", "Weighted mean reach, input pixels", "{:.1f}"),
    ):
        lines.append(f"### {title}\n")
        lines.append("| size | n | " + " | ".join(f"layer {i}" for i in range(num_layers)) + " |")
        lines.append("| --- | ---: | " + " | ".join("---:" for _ in range(num_layers)) + " |")
        for name in names:
            rows = records if name == "all" else [r for r in records if r["bucket"] == name]
            if not rows:
                continue
            values = np.array([r[key] for r in rows]).mean(0)
            lines.append(f"| {name} | {len(rows)} | " + " | ".join(fmt.format(v) for v in values) + " |")
        lines.append("")
    lines.append("### Share of attention weight per feature level\n")
    lines.append("| size | n | layer | " + " | ".join(f"stride {s}" for s in strides) + " |")
    lines.append("| --- | ---: | ---: | " + " | ".join("---:" for _ in strides) + " |")
    for name in names:
        rows = records if name == "all" else [r for r in records if r["bucket"] == name]
        if not rows:
            continue
        for layer in range(num_layers):
            values = np.array([r["level_share"][layer] for r in rows]).mean(0)
            lines.append(f"| {name} | {len(rows)} | {layer} | " + " | ".join(f"{v:.0%}" for v in values) + " |")
    lines.append("")
    for key, title, fmt in (
        ("half_cells", "Half-size of the box the layer starts from, in cells of the level", "{:.2f}"),
        ("spread", "Weighted spread of a head's points of the level, in cells (largest Chebyshev distance)", "{:.2f}"),
        ("collapsed", "Share of the level's weight on heads whose points all lie within one cell", "{:.0%}"),
    ):
        lines.append(f"### {title}\n")
        lines.append("| size | n | layer | " + " | ".join(f"stride {s}" for s in strides) + " |")
        lines.append("| --- | ---: | ---: | " + " | ".join("---:" for _ in strides) + " |")
        for name in names:
            rows = records if name == "all" else [r for r in records if r["bucket"] == name]
            if not rows:
                continue
            for layer in range(num_layers):
                values = np.array([r[key][layer] for r in rows]).mean(0)
                lines.append(f"| {name} | {len(rows)} | {layer} | " + " | ".join(fmt.format(v) for v in values) + " |")
        lines.append("")
    return "\n".join(lines)


def records_of(sample, iou_threshold, strides):
    records = []
    num_layers = len(sample.attention)
    for g, q, _ in sample.match(iou_threshold):
        stats = [object_stats(sample, layer, q, g) for layer in range(num_layers)]
        cells = [cell_stats(sample, layer, q, strides) for layer in range(num_layers)]
        records.append(
            {
                "bucket": bucket_of(sample.gt_size(g)),
                "half_cells": [c["half_cells"] for c in cells],
                "spread": [c["spread"] for c in cells],
                "collapsed": [c["collapsed"] for c in cells],
                "inside": [s["inside"] for s in stats],
                "within_1": [s["within_1"] for s in stats],
                "reach_units": [s["reach_units"] for s in stats],
                "reach_px": [s["reach_px"] for s in stats],
                "level_share": [s["level_share"] for s in stats],
                "rel": [s["rel"].numpy() for s in stats],
                "weights": [s["weights"].numpy() for s in stats],
            }
        )
    return records


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", help="output directory (default <run>/attention)")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(args.run, "attention")
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = pick_checkpoint(args.run, args.checkpoint)
    print(f"checkpoint {checkpoint}")

    cfg = YAMLConfig(os.path.join(args.run, "config.yml"))
    model = load_model(cfg, checkpoint, args.device)
    decoder = find_decoder(model)
    strides = list(model.decoder.feat_strides)
    loader = cfg.val_dataloader
    dataset, collate = loader.dataset, loader.collate_fn
    names = dict(getattr(dataset, "CATEGORIES", []))
    attn = decoder.layers[0].cross_attn
    print(
        f"decoder: {decoder.num_layers} layers, {attn.num_heads} heads, points per level {attn.num_points_list}, strides {strides}, offset_scale {attn.offset_scale}, min_sample_cells {attn.min_sample_cells}"
    )

    index = resolve_image(dataset, args.image)
    sample = sample_with_attention(dataset, collate, index, model, decoder, args.device)
    matched = sorted(sample.match(args.match_iou), key=lambda m: sample.gt_size(m[0]))
    picks = (
        np.unique(np.linspace(0, len(matched) - 1, min(args.objects, len(matched))).round().astype(int))
        if matched
        else []
    )
    for k, i in enumerate(picks):
        g, q, iou = matched[i]
        object_figure(sample, g, q, strides, os.path.join(out_dir, f"object_{k}.png"), names)
        print(
            f"  object_{k}.png: gt {g} ({names.get(int(sample.gt_labels[g]), '?')}, {sample.gt_size(g):.0f}px), query {q}, final IoU {iou:.2f}"
        )

    if args.num_images > 0:
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(dataset)), min(args.num_images, len(dataset)))
        records = []
        for i, idx in enumerate(indices):
            s = sample if idx == index else sample_with_attention(dataset, collate, idx, model, decoder, args.device)
            records += records_of(s, args.match_iou, strides)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(indices)} images, {len(records)} objects")
        if records:
            trend_figure(records, decoder.num_layers, os.path.join(out_dir, "trend.png"))
            text = (
                f"# Deformable cross-attention of `{os.path.relpath(checkpoint).replace(os.sep, '/')}`\n\n"
                f"{len(records)} ground-truth objects of {len(indices)} validation images, each followed through its detection's query "
                f"(the highest-scoring query whose final box reaches IoU {args.match_iou}). Every layer samples "
                f"{attn.num_heads} heads x {sum(attn.num_points_list)} points; weights are softmaxed per head and pooled over heads. "
                f"Reach is the Chebyshev distance of a sampling point from the box centre, in box sizes or input pixels. "
                f"The per-level tables measure the box a layer starts from (the proposal for layer 0, else the previous "
                f"layer's box) and each head's {attn.num_points_list[0]} points of a level in cells of that level; points "
                f"within one cell read the same bilinear neighbourhood. Sizes are square-root areas in original pixels."
                f"\n\n![trend](trend.png)\n\n" + trend_table(records, decoder.num_layers, strides)
            )
            with open(os.path.join(out_dir, "trend.md"), "w", encoding="utf-8") as f:
                f.write(text)
            print(text)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
