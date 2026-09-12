"""
How well the counting head of a ``query_budget: bucket`` run sorts the validation images into
their count buckets, and what budgets it hands out: the confusion of ground-truth against
chosen bucket (the 0.9-quantile rule the model uses at inference, and the plain argmax), how
often the budget covers the image's objects, and the queries spent against fixed alternatives.
    python tools/analysis/count_probe.py outputs/ablation/aitod_s_2_budget/<run>
    python tools/analysis/count_probe.py <run> --checkpoint last.pth --limit 2000
"""

import argparse
import contextlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fdr_refinement import load_model, pick_checkpoint  # noqa: E402

from src.core import YAMLConfig  # noqa: E402
from src.zoo.dome.dfine_decoder import DFINETransformer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run directory (config.yml and checkpoints), or a config file with --random")
    parser.add_argument("--checkpoint", help="checkpoint file (default: best_stg2, best_stg1 or last)")
    parser.add_argument("--limit", type=int, default=0, help="images to probe (default: the whole split)")
    parser.add_argument("--random", action="store_true", help="an untrained model, to check the probe runs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = args.run if args.run.endswith(".yml") else os.path.join(args.run, "config.yml")
    cfg = YAMLConfig(config)
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    if args.random:
        model = cfg.model.to(args.device).eval()
    else:
        model = load_model(cfg, pick_checkpoint(args.run, args.checkpoint), args.device)
    decoder = next(m for m in model.modules() if isinstance(m, DFINETransformer))
    assert decoder.query_budget == "bucket", "the run has no counting head (query_budget is not 'bucket')"
    edges, budgets = decoder.count_edges, decoder.count_budgets
    labels = [f"<{edges[0]}"] + [f"{a}-{b}" for a, b in zip(edges, edges[1:])] + [f">={edges[-1]}"]

    logits_all, counts = [], []
    decoder.count_head.register_forward_hook(lambda m, i, o: logits_all.append(o.detach().float().cpu()))
    loader = cfg.val_dataloader
    seen = 0
    with torch.no_grad(), contextlib.redirect_stdout(open(os.devnull, "w")):
        for samples, targets in loader:
            model(samples.to(args.device))
            counts.extend(len(t["labels"]) for t in targets)
            seen += len(targets)
            if args.limit and seen >= args.limit:
                break
    logits = torch.cat(logits_all)[: len(counts)]
    counts = np.array(counts)
    gt = np.digitize(counts, edges, right=False)  # 100 falls in bucket 1, as torch.bucketize(right=True)
    cdf = logits.softmax(-1).cumsum(-1)
    chosen = (cdf >= decoder.count_quantile - 1e-6).int().argmax(-1).numpy()
    argmax = logits.argmax(-1).numpy()
    budget = np.array(budgets)[chosen]

    k = len(budgets)
    print(f"{len(counts)} images; buckets {labels}; budgets {budgets}; quantile {decoder.count_quantile}")
    print(
        f"\n{'GT bucket':>10s} {'n':>6s} {'argmax acc':>10s} {'chosen acc':>10s} {'under':>6s} {'over':>6s} {'covered':>8s} {'mean budget':>11s} {'mean count':>10s}"
    )
    for b in range(k):
        m = gt == b
        if not m.any():
            print(f"{labels[b]:>10s} {0:6d}")
            continue
        print(
            f"{labels[b]:>10s} {m.sum():6d} {100 * (argmax[m] == b).mean():9.1f}% {100 * (chosen[m] == b).mean():9.1f}%"
            f" {100 * (chosen[m] < b).mean():5.1f}% {100 * (chosen[m] > b).mean():5.1f}%"
            f" {100 * (budget[m] >= counts[m]).mean():7.1f}% {budget[m].mean():11.0f} {counts[m].mean():10.1f}"
        )
    print(
        f"{'all':>10s} {len(counts):6d} {100 * (argmax == gt).mean():9.1f}% {100 * (chosen == gt).mean():9.1f}%"
        f" {100 * (chosen < gt).mean():5.1f}% {100 * (chosen > gt).mean():5.1f}% {100 * (budget >= counts).mean():7.1f}%"
        f" {budget.mean():11.0f} {counts.mean():10.1f}"
    )
    print("\nconfusion, rows GT bucket, columns chosen bucket:")
    print(f"{'':>10s}" + "".join(f"{l:>9s}" for l in labels))
    for b in range(k):
        row = [(gt == b) & (chosen == c) for c in range(k)]
        print(f"{labels[b]:>10s}" + "".join(f"{int(r.sum()):9d}" for r in row))
    fixed = {n: 100 * (counts <= n).mean() for n in (budgets[0], budgets[-1])}
    print(
        f"\nqueries per image: budget {budget.mean():.0f} on average against fixed {budgets[0]} / {budgets[-1]};"
        f" images covered: budget {100 * (budget >= counts).mean():.1f}%, fixed {budgets[0]}: {fixed[budgets[0]]:.1f}%,"
        f" fixed {budgets[-1]}: {fixed[budgets[-1]]:.1f}%"
    )


if __name__ == "__main__":
    main()
