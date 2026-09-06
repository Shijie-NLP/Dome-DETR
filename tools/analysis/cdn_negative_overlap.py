"""
Do the contrastive-denoising (CDN) negatives of one ground-truth box land on *another*
ground-truth box? CDN queries are sampled exactly as training does and, for positives and
negatives, compared with their own source GT and with the other GT they overlap most:

* ``other>=thr``: the query overlaps some other GT at least ``thr``.
* ``other>own``: the query is closer to some other GT than to its own source.
* ``beats pos``: negatives only; take the other GT the negative overlaps most and compare with
  that GT's own positive in the same CDN group. The negative is the better box for it.
* Hungarian: match every group's positives *and* negatives to the GT with the training matcher's
  box cost (5 L1 + 2 GIoU, no class term since dn queries carry no scores) and record what each
  GT gets: its own positive, its own negative, or another GT's positive / negative.

Ground truth comes from the Hub through the repo's dataset classes, filtered exactly like a
training target (ignore regions and degenerate boxes out).

Single image (bucketed by object size, optional IoU histograms):
    python tools/analysis/cdn_negative_overlap.py visdrone --image 0000059_01886_d_0000114 --plot out.png
Dataset sample (bucketed by the number of GT boxes per image, with the GT-GT overlap baseline):
    python tools/analysis/cdn_negative_overlap.py aitod --split train --num-images 500
"""

import argparse
import os
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.dataset import AITODDetection, VisDroneDetection, VOCDetection  # noqa: E402
from src.misc.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou  # noqa: E402
from src.zoo.dome.denoising import get_contrastive_denoising_training_group  # noqa: E402

DATASETS = {"visdrone": VisDroneDetection, "aitod": AITODDetection, "voc": VOCDetection}
SIZE_BUCKETS = [("tiny <16px", 0, 16), ("small 16-32px", 16, 32), ("medium 32-96px", 32, 96), ("large >96px", 96, 1e9)]
COUNT_BUCKETS = [("1-50 GT", 1, 50), ("51-200 GT", 51, 200), ("201-500 GT", 201, 500), (">500 GT", 501, 10**9)]
COST_BBOX, COST_GIOU = 5.0, 2.0  # the training matcher's box terms
MATCH_KINDS = ("own pos", "own neg", "other pos", "other neg")


def load_gt(dataset, index):
    """
    The training target of row ``index`` without decoding the image: boxes (normalized cxcywh),
    labels, and sqrt(area) in pixels.
    """
    row = dataset.hf_meta[index]
    w, h = row["width"], row["height"]
    boxes, labels, iscrowd = dataset.parse_objects(row["objects"])
    boxes = boxes.clone()
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)
    keep = (iscrowd == 0) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, labels = boxes[keep], labels[keep]
    bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    cxcywh = torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 1] + boxes[:, 3]) / 2 / h, bw / w, bh / h], 1)
    return cxcywh, labels, torch.sqrt(bw * bh)


def sample_cdn(boxes, labels, num_classes, num_denoising, box_noise_scale, label_noise_ratio):
    """One CDN draw. Returns dn boxes (cxcywh, [N_dn, 4]), the source-GT index and is_negative of every dn query."""
    targets = [{"labels": labels, "boxes": boxes}]
    class_embed = torch.nn.Embedding(num_classes + 1, 8, padding_idx=num_classes)
    _, dn_bbox_unact, _, dn_meta = get_contrastive_denoising_training_group(
        targets,
        num_classes,
        300,
        class_embed,
        num_denoising=num_denoising,
        label_noise_ratio=label_noise_ratio,
        box_noise_scale=box_noise_scale,
    )
    dn_cxcywh = torch.sigmoid(dn_bbox_unact[0])
    n_gt = len(labels)
    idx = torch.arange(dn_cxcywh.shape[0])
    source = idx % n_gt  # layout: [group0: n_gt positives, n_gt negatives][group1: ...]
    is_negative = (idx // n_gt) % 2 == 1
    return dn_cxcywh, source, is_negative


def hungarian(dn_cxcywh, gt_cxcywh, n_gt):
    """
    Per CDN group, the training matcher's box cost between its 2 * n_gt queries and the GT, then
    a Hungarian assignment. Returns, per GT and group, what it was matched to as an index into
    ``MATCH_KINDS``.
    """
    gt_xyxy = box_cxcywh_to_xyxy(gt_cxcywh)
    kinds = []
    for g in range(dn_cxcywh.shape[0] // (2 * n_gt)):
        q = dn_cxcywh[g * 2 * n_gt : (g + 1) * 2 * n_gt]
        cost = COST_BBOX * torch.cdist(q, gt_cxcywh, p=1) - COST_GIOU * generalized_box_iou(
            box_cxcywh_to_xyxy(q), gt_xyxy
        )
        rows, cols = linear_sum_assignment(torch.nan_to_num(cost, nan=1.0).numpy())
        matched = torch.empty(n_gt, dtype=torch.int64)
        matched[cols] = torch.from_numpy(rows)  # every GT gets one query: 2 * n_gt >= n_gt
        own = matched % n_gt == torch.arange(n_gt)
        negative = matched >= n_gt
        kinds.append((~own).long() * 2 + negative.long())
    return torch.cat(kinds)


def measure(boxes, labels, num_classes, repeat, num_denoising, box_noise_scale):
    """
    ``repeat`` CDN draws on one image, concatenated. Per dn query: ``own`` (IoU with the source
    GT), ``other`` / ``other_arg`` (best IoU with any other GT and which), ``other_pos`` (the own
    IoU of that GT's positive in the same group), ``neg`` and ``src``; and per GT and group,
    ``match``, what the Hungarian assignment gave it (see ``hungarian``).
    """
    gt_xyxy = box_cxcywh_to_xyxy(boxes)
    n_gt = len(labels)
    cols = {k: [] for k in ("own", "other", "other_arg", "other_pos", "neg", "src", "match")}
    for _ in range(repeat):
        dn_cxcywh, source, is_negative = sample_cdn(boxes, labels, num_classes, num_denoising, box_noise_scale, 0.0)
        idx = torch.arange(len(source))
        iou, _ = box_iou(box_cxcywh_to_xyxy(dn_cxcywh), gt_xyxy)  # [N_dn, N_gt]
        own = iou[idx, source]
        iou[idx, source] = -1
        other, other_arg = iou.max(dim=1)
        pos_of_other = (idx // (2 * n_gt)) * 2 * n_gt + other_arg  # that GT's positive, same group
        cols["own"].append(own)
        cols["other"].append(other)
        cols["other_arg"].append(other_arg)
        cols["other_pos"].append(own[pos_of_other])
        cols["neg"].append(is_negative)
        cols["src"].append(source)
        cols["match"].append(hungarian(dn_cxcywh, boxes, n_gt))  # per GT and group, not per query
    return {k: torch.cat(v) for k, v in cols.items()}


def counts(m, mask, thr):
    """Sums of the per-query indicators over ``mask``; add across images for dataset-level rates."""
    other, own = m["other"][mask], m["own"][mask]
    return {
        "n": int(mask.sum()),
        "other_thr": int((other >= thr).sum()),
        "other_gt_own": int((other > own).sum()),
        "beats": int((other > m["other_pos"][mask]).sum()),
    }


def match_counts(m):
    """How many (GT, group) pairs the Hungarian assignment gave each kind of query."""
    return {"n": len(m["match"]), **{k: int((m["match"] == i).sum()) for i, k in enumerate(MATCH_KINDS)}}


def match_row(c):
    return " | ".join(pct(c[k], c["n"]) for k in MATCH_KINDS)


def add(acc, c):
    for k, v in c.items():
        acc[k] = acc.get(k, 0) + v
    return acc


def pct(num, den):
    return f"{num / den * 100:5.1f}%" if den else "    -"


def run_image(dataset, num_classes, args):
    ids = dataset.hf_meta["id"]
    if args.image not in ids:
        raise SystemExit(f"{args.image} is not in {dataset}")
    boxes, labels, sizes = load_gt(dataset, ids.index(args.image))
    n_gt = len(labels)
    if n_gt < 2:
        raise SystemExit(f"{args.image} has {n_gt} GT boxes; nothing to overlap with")
    print(f"image {args.image}: {n_gt} GT boxes, sqrt(area) median {sizes.median():.1f}px")
    m = measure(boxes, labels, num_classes, args.repeat, args.num_denoising, args.box_noise_scale)
    neg, src_size = m["neg"], sizes[m["src"]]
    print(f"{int((~neg).sum())} positive and {int(neg.sum())} negative CDN queries over {args.repeat} draws\n")

    def row(mask, with_beats):
        c = counts(m, mask, args.thr)
        o, t = m["own"][mask], m["other"][mask]
        cells = [
            f"{c['n']:6d}",
            f"{o.mean():.3f}",
            f"{o.median():.3f}",
            f"{t.mean():.3f}",
            f"{t.median():.3f}",
            pct(c["other_thr"], c["n"]),
            pct(c["other_gt_own"], c["n"]),
        ]
        if with_beats:
            cells.append(pct(c["beats"], c["n"]))
        return " | ".join(cells)

    header = f"n | own IoU mean | own med | other IoU mean | other med | other>={args.thr} | other>own"
    for name, mask, with_beats in [("POSITIVES", ~neg, False), ("NEGATIVES", neg, True)]:
        print(f"{name} (all sizes)\n{header}{' | beats pos' if with_beats else ''}\n{row(mask, with_beats)}")
        for bname, lo, hi in SIZE_BUCKETS:
            bmask = mask & (src_size >= lo) & (src_size < hi)
            if bmask.any():
                print(f"  {bname:16s} {row(bmask, with_beats)}")
        print()

    beats = neg & (m["other"] > m["other_pos"])
    if beats.any():
        same_class = labels[m["src"][beats]] == labels[m["other_arg"][beats]]
        print(
            "negatives that beat the other GT's positive: "
            f"same class as that GT in {same_class.float().mean() * 100:.1f}%\n"
        )

    print(
        f"HUNGARIAN over each group's positives + negatives: what every GT is matched to ({len(m['match'])} GT x groups)"
    )
    print(f"{'':16s} " + " | ".join(MATCH_KINDS))
    print(f"{'all sizes':16s} {match_row(match_counts(m))}")
    gt_size = sizes.repeat(len(m["match"]) // n_gt)
    for bname, lo, hi in SIZE_BUCKETS:
        bmask = (gt_size >= lo) & (gt_size < hi)
        if bmask.any():
            print(f"  {bname:16s} {match_row(match_counts({'match': m['match'][bmask]}))}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bins = np.linspace(0, 1, 41)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(m["own"][~neg].numpy(), bins, alpha=0.6, label="positive: IoU with own GT")
        axes[0].hist(m["own"][neg].numpy(), bins, alpha=0.6, label="negative: IoU with own GT")
        axes[0].set_title("IoU with the source GT")
        axes[1].hist(m["other"][~neg].numpy(), bins, alpha=0.6, label="positive: best IoU with another GT")
        axes[1].hist(m["other"][neg].numpy(), bins, alpha=0.6, label="negative: best IoU with another GT")
        axes[1].set_title("best IoU with any OTHER GT")
        for ax in axes:
            ax.legend()
            ax.set_yscale("log")
        fig.suptitle(f"{args.image}: {n_gt} GT, box_noise_scale={args.box_noise_scale}, {args.repeat} draws")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"saved {args.plot}")


def run_dataset(dataset, num_classes, args):
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=min(args.num_images, len(dataset)), replace=False)

    rows = []  # per image: n_gt, median size, GT-GT baseline, positive counts, negative counts
    skipped = 0
    for index in indices.tolist():
        boxes, labels, sizes = load_gt(dataset, index)
        if len(labels) < 2:  # no other GT to overlap with
            skipped += 1
            continue
        gt_xyxy = box_cxcywh_to_xyxy(boxes)
        gg, _ = box_iou(gt_xyxy, gt_xyxy)
        gg.fill_diagonal_(-1)
        gt_overlap = (gg.max(dim=1).values >= args.thr).float().mean().item()
        m = measure(boxes, labels, num_classes, args.repeat, args.num_denoising, args.box_noise_scale)
        pos, neg = counts(m, ~m["neg"], args.thr), counts(m, m["neg"], args.thr)
        rows.append((len(labels), sizes.median().item(), gt_overlap, pos, neg, match_counts(m)))

    print(
        f"{dataset}\n{len(rows)} images ({skipped} with fewer than 2 GT skipped), "
        f"box_noise_scale={args.box_noise_scale}, thr={args.thr}, {args.repeat} draws each\n"
    )
    print(
        f"{'bucket':12s} | imgs | med size | GT-GT>={args.thr} | pos other>={args.thr} | pos other>own | "
        f"neg other>={args.thr} | neg other>own | neg beats pos"
    )

    def report(name, sel):
        if not sel:
            return
        pos, neg = {}, {}
        for *_, p, n, _ in sel:
            add(pos, p)
            add(neg, n)
        print(
            f"{name:12s} | {len(sel):4d} | {np.median([r[1] for r in sel]):6.1f}px | "
            f"{np.mean([r[2] for r in sel]) * 100:6.1f}% | "
            f"{pct(pos['other_thr'], pos['n'])} | {pct(pos['other_gt_own'], pos['n'])} | "
            f"{pct(neg['other_thr'], neg['n'])} | {pct(neg['other_gt_own'], neg['n'])} | {pct(neg['beats'], neg['n'])}"
        )

    def report_match(name, sel):
        if not sel:
            return
        c = {}
        for *_, mc in sel:
            add(c, mc)
        print(f"{name:12s} | {len(sel):4d} | {c['n']:8d} | {match_row(c)}")

    buckets = [(name, [r for r in rows if lo <= r[0] <= hi]) for name, lo, hi in COUNT_BUCKETS] + [("all", rows)]
    for name, sel in buckets:
        report(name, sel)

    print("\nHUNGARIAN over each group's positives + negatives: what every GT is matched to")
    print(f"{'bucket':12s} | imgs | GT x grp | " + " | ".join(MATCH_KINDS))
    for name, sel in buckets:
        report_match(name, sel)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--split", default="train", help="a Hub split, or splits joined with '+'")
    parser.add_argument("--image", default=None, help="one image id; omitted, a random sample of the split is used")
    parser.add_argument("--num-images", type=int, default=500, help="sample size in dataset mode")
    parser.add_argument(
        "--repeat", type=int, default=None, help="CDN draws per image (default 20 for an image, 2 for a dataset)"
    )
    parser.add_argument("--num-denoising", type=int, default=100)
    parser.add_argument("--box-noise-scale", type=float, default=1.0)
    parser.add_argument("--thr", type=float, default=0.3, help="IoU threshold for the other>=thr columns")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", default=None, help="single image only: png path for the IoU histograms")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    dataset = DATASETS[args.dataset](split=args.split)
    num_classes = max(i for i, _ in dataset.CATEGORIES) + 1  # labels index the class embedding
    if args.image:
        args.repeat = args.repeat or 20
        run_image(dataset, num_classes, args)
    else:
        args.repeat = args.repeat or 2
        run_dataset(dataset, num_classes, args)


if __name__ == "__main__":
    main()
