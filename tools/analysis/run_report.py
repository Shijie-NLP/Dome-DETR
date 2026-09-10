"""
Summarize a training run from what ``train.py`` leaves in its output directory: ``config.yml``,
the per-epoch ``log.txt``, ``console.log``, the checkpoints and the COCOeval dumps under
``eval/``. Writes ``RESULTS.md`` and ``curves.png`` next to them, so one look tells how the run
was set up, whether it finished, how far it got, which checkpoint holds the best epoch and
how the metrics, losses, learning rate and epoch times moved.

    python tools/analysis/run_report.py outputs/dfine_s_visdrone/2026-09-09_17-13-58
    python tools/analysis/run_report.py outputs/*/*            # every run, one report each

The per-class table needs torch (the ``eval/*.pth`` dumps are torch pickles) and is skipped
without it; everything else is plain text parsing.
"""

import argparse
import datetime
import json
import math
import os
import re
import sys
import time

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# the entries of ``coco_eval_bbox`` per evaluator (``summarize`` in src/data/dataset/*_eval.py)
METRIC_LABELS = {
    "VisDroneEvaluator": [
        "AP",
        "AP50",
        "AP75",
        "AP_s",
        "AP_m",
        "AP_l",
        "AR@1",
        "AR@10",
        "AR@100",
        "AR@500",
        "AR_s",
        "AR_m",
    ],
    "AITODEvaluator": [
        "AP",
        "AP50",
        "AP75",
        "AP_vt",
        "AP_t",
        "AP_s",
        "AP_m",
        "AR@1",
        "AR@100",
        "AR@1500",
        "AR_vt",
        "AR_t",
        "AR_s",
        "AR_m",
    ],
    "CocoEvaluator": ["AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l", "AR@1", "AR@10", "AR@100", "AR_s", "AR_m", "AR_l"],
}

# fixed-order categorical palette, plus the recessive chart furniture
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e6e5e1"


# --------------------------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------------------------


def read_log(path):
    """The epochs of ``log.txt`` by epoch number; a resumed run may log an epoch twice, the last wins."""
    epochs = {}
    if not os.path.exists(path):
        return []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" in row:
                epochs[row["epoch"]] = row
    return [epochs[e] for e in sorted(epochs)]


def parse_duration(text):
    """``h:mm:ss`` or ``d day(s), h:mm:ss`` in seconds."""
    days = 0
    m = re.match(r"(\d+) days?, (.*)", text)
    if m:
        days, text = int(m.group(1)), m.group(2)
    h, mi, s = (int(x) for x in text.split(":"))
    return days * 86400 + h * 3600 + mi * 60 + s


def read_console(path):
    """What ``console.log`` says beyond ``log.txt``: launches, timing, memory, stage-2 events, trouble."""
    info = {
        "launches": [],  # (timestamp, command line)
        "flops": None,
        "params": None,
        "iters_per_epoch": None,
        "train_time": {},  # epoch -> seconds
        "train_s_it": {},  # epoch -> seconds per iteration
        "eval_time": {},  # epoch -> seconds
        "max_mem": 0,
        "reloads": [],  # (epoch, decay)
        "patience": [],  # (epoch, n, patience)
        "training_time": None,
        "tracebacks": 0,
        "warnings": set(),
    }
    if not os.path.exists(path):
        return info
    epoch = None
    with open(path, errors="replace") as f:
        for line in f:
            m = re.match(r"===== (\S+)\s+(.*)$", line)
            if m:
                info["launches"].append((m.group(1), m.group(2).strip()))
                continue
            m = re.search(r"Model FLOPs:\s*([\d.]+) GFLOPs.*Params:\s*(\d+)", line)
            if m:
                info["flops"], info["params"] = float(m.group(1)), int(m.group(2))
                continue
            m = re.match(r"Epoch: \[(\d+)\]\s+\[\s*\d+/(\d+)\]", line)
            if m:
                epoch = int(m.group(1))
                info["iters_per_epoch"] = int(m.group(2))
                mm = re.search(r"max mem: (\d+)", line)
                if mm:
                    info["max_mem"] = max(info["max_mem"], int(mm.group(1)))
                continue
            m = re.match(r"Epoch: \[(\d+)\] Total time: (\S+) \(([\d.]+) s / it\)", line)
            if m:
                epoch = int(m.group(1))
                info["train_time"][epoch] = parse_duration(m.group(2))
                info["train_s_it"][epoch] = float(m.group(3))
                continue
            m = re.match(r"Test: Total time: (\S+)", line)
            if m and epoch is not None:
                info["eval_time"][epoch] = parse_duration(m.group(1))
                continue
            m = re.search(r"Reload .* at epoch (\d+)", line)
            if m:
                info["reloads"].append([int(m.group(1)), None])
                continue
            m = re.search(r"Refresh EMA at epoch (\d+) with decay ([\d.]+)", line)
            if m and info["reloads"] and info["reloads"][-1][0] == int(m.group(1)):
                info["reloads"][-1][1] = float(m.group(2))
                continue
            m = re.search(r"Tolerate undesirable result for patience: (\d+) / (\d+)", line)
            if m and epoch is not None:
                info["patience"].append((epoch, int(m.group(1)), int(m.group(2))))
                continue
            m = re.match(r"Training time (.*)$", line)
            if m:
                info["training_time"] = parse_duration(m.group(1).strip())
                continue
            if line.startswith("Traceback"):
                info["tracebacks"] += 1
            m = re.search(r"(\w*Warning): (.*)$", line)
            if m:
                info["warnings"].add(f"{m.group(1)}: {m.group(2).strip()[:120]}")
    return info


def load_eval(path):
    """A COCOeval ``.eval`` dump as per-class (AP, AP50, AR@maxDets) by category id, or None."""
    try:
        import numpy as np
        import torch

        e = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as err:  # noqa: BLE001 - torch missing, or an older dump
        print(f"  skip {path}: {err}")
        return None

    def mean_valid(x):
        x = x[x > -1]
        return float(x.mean()) if x.size else float("nan")

    precision, recall = np.asarray(e["precision"]), np.asarray(e["recall"])  # [T,R,K,A,M], [T,K,A,M]
    per_class = {}
    for k, cat in enumerate(e["params"].catIds):
        per_class[int(cat)] = (
            mean_valid(precision[:, :, k, 0, -1]),
            mean_valid(precision[0, :, k, 0, -1]),
            mean_valid(recall[:, k, 0, -1]),
        )
    return per_class


def category_names(cfg):
    """Category id -> name through the dataset class the run validated on, when importable."""
    dataset_type = (cfg.get("val_dataloader") or {}).get("dataset", {}).get("type")
    try:
        import src.data.dataset as datasets

        return dict(getattr(datasets, dataset_type).CATEGORIES)
    except Exception:  # noqa: BLE001 - no torch, or a dataset without CATEGORIES
        return {}


# --------------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------------


class Run:
    def __init__(self, directory):
        self.dir = directory
        self.name = os.path.relpath(directory).replace("\\", "/")
        with open(os.path.join(directory, "config.yml")) as f:
            self.cfg = yaml.safe_load(f)
        self.log = read_log(os.path.join(directory, "log.txt"))
        self.console = read_console(os.path.join(directory, "console.log"))

        self.epochs = [row["epoch"] for row in self.log]
        self.total_epochs = self.cfg.get("epoches")
        collate = (self.cfg.get("train_dataloader") or {}).get("collate_fn") or {}
        self.stage2_start = collate.get("stop_epoch")
        evaluator = (self.cfg.get("evaluator") or {}).get("type", "CocoEvaluator")
        self.metric_key = next(
            (k for k in ("test_coco_eval_bbox", "test_coco_eval_masks") if self.log and k in self.log[0]), None
        )
        n_stats = len(self.log[0][self.metric_key]) if self.metric_key else 0
        labels = METRIC_LABELS.get(evaluator, METRIC_LABELS["CocoEvaluator"])
        self.labels = labels if len(labels) == n_stats else [f"stat{i}" for i in range(n_stats)]
        self.evaluator = evaluator

    # -- metrics ------------------------------------------------------------------------------

    def stats(self, row):
        return row[self.metric_key] if self.metric_key else []

    def ap(self, row):
        return self.stats(row)[0] if self.metric_key else float("nan")

    def series(self, key):
        """``key`` per logged epoch, None where an epoch lacks it."""
        return [row.get(key) for row in self.log]

    def metric_series(self, label):
        i = self.labels.index(label)
        return [self.stats(row)[i] for row in self.log]

    def best(self, rows=None):
        """The first row with the highest AP, as the solver's strict ``>`` keeps it."""
        rows = self.log if rows is None else rows
        if not rows:
            return None
        return max(rows, key=lambda r: (self.ap(r), -r["epoch"]))

    def stage_rows(self, stage):
        if self.stage2_start is None:
            return self.log if stage == 1 else []
        return [r for r in self.log if (r["epoch"] >= self.stage2_start) == (stage == 2)]

    def checkpoint_epochs(self):
        """Which epoch each checkpoint file holds, from how the solver writes them."""
        held = {}
        best = self.best()
        best1 = self.best(self.stage_rows(1))
        if best1 is not None:
            held["best_stg1.pth"] = best1["epoch"]
        if best is not None and self.stage2_start is not None and best["epoch"] >= self.stage2_start:
            held["best_stg2.pth"] = best["epoch"]
        stage1 = self.stage_rows(1)
        if stage1:
            held["last.pth"] = stage1[-1]["epoch"]
        for name in os.listdir(self.dir):
            m = re.match(r"checkpoint(\d+)\.pth$", name)
            if m:
                held[name] = int(m.group(1))
        return held

    # -- status -------------------------------------------------------------------------------

    def status(self):
        c = self.console
        last = self.epochs[-1] if self.epochs else None
        if c["tracebacks"]:
            state = "crashed"
        elif c["training_time"] is not None:
            state = "finished"
        elif last is not None and self.total_epochs and last >= self.total_epochs - 1:
            state = "finished (no summary line, the process may have been cut after the last epoch)"
        else:
            log_path = os.path.join(self.dir, "log.txt")
            age = time.time() - os.path.getmtime(log_path) if os.path.exists(log_path) else float("inf")
            state = "running" if age < 3 * self.mean_epoch_seconds() + 600 else "stopped"
        return state

    def mean_epoch_seconds(self):
        times = list(self.console["train_time"].values())
        evals = list(self.console["eval_time"].values())
        return (sum(times) / len(times) if times else 0) + (sum(evals) / len(evals) if evals else 0)

    def wall_seconds(self):
        if self.console["training_time"] is not None:
            return self.console["training_time"]
        return sum(self.console["train_time"].values()) + sum(self.console["eval_time"].values())


# --------------------------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------------------------


def fmt_pct(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def fmt_num(x, digits=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}g}" if abs(x) < 1e-2 or abs(x) >= 1e4 else f"{x:.{digits}f}"
    return str(x)


def fmt_duration(seconds):
    if seconds is None:
        return "-"
    return str(datetime.timedelta(seconds=int(seconds)))


def table(headers, rows, align=None):
    align = align or ["---"] + ["---:"] * (len(headers) - 1)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def setup_rows(run):
    cfg, c = run.cfg, run.console
    model = cfg.get("model")
    parts = cfg.get(model) or {}
    rows = [("config", f"`{cfg.get('config', '-')}`"), ("seed", cfg.get("seed", "-"))]
    for launched, command in c["launches"]:
        rows.append(("launched", f"{launched} `{command}`"))
    rows.append(("model", model))
    for role in ("backbone", "encoder", "decoder"):
        name = parts.get(role)
        if not name:
            continue
        sub = cfg.get(name) or {}
        picked = {
            k: sub[k]
            for k in ("name", "num_queries", "num_layers", "num_denoising", "use_mwas", "mwas_window_size", "defe_type")
            if k in sub
        }
        detail = ", ".join(f"{k}={v}" for k, v in picked.items())
        rows.append((role, f"{name}" + (f" ({detail})" if detail else "")))
    if c["params"] or c["flops"]:
        rows.append(
            (
                "size",
                f"{c['params'] / 1e6:.2f} M params, {c['flops']:.0f} GFLOPs"
                if c["flops"]
                else f"{c['params'] / 1e6:.2f} M params",
            )
        )
    train_ds, val_ds = get(cfg, "train_dataloader", "dataset") or {}, get(cfg, "val_dataloader", "dataset") or {}
    rows.append(
        (
            "train data",
            f"{train_ds.get('type', '-')} `{train_ds.get('split', '-')}`, batch {get(cfg, 'train_dataloader', 'total_batch_size', default='-')}",
        )
    )
    rows.append(
        (
            "val data",
            f"{val_ds.get('type', '-')} `{val_ds.get('split', '-')}`, batch {get(cfg, 'val_dataloader', 'total_batch_size', default='-')}, {run.evaluator}",
        )
    )
    collate = get(cfg, "train_dataloader", "collate_fn") or {}
    train_aug = ", ".join(
        f"{k}={collate[k]}" for k in ("base_size", "base_size_repeat", "stop_epoch", "mwas_window_size") if k in collate
    )
    if train_aug:
        rows.append(("train batches", train_aug))
    val_ops = [
        op.get("type") + (f" {op['size']}" if "size" in op else "")
        for op in (get(val_ds, "transforms", "ops") or [])
        if op.get("type") not in ("ConvertPILImage", "ConvertBoxes")
    ]
    if val_ops:
        rows.append(("val transforms", ", ".join(val_ops)))
    opt, sched = cfg.get("optimizer") or {}, cfg.get("lr_scheduler") or {}
    rows.append(
        (
            "epochs",
            f"{run.total_epochs}" + (f", stage 2 from {run.stage2_start}" if run.stage2_start is not None else ""),
        )
    )
    rows.append(("optimizer", f"{opt.get('type', '-')} lr={opt.get('lr', '-')} wd={opt.get('weight_decay', '-')}"))
    rows.append(
        ("schedule", f"{sched.get('type', '-')} {', '.join(f'{k}={v}' for k, v in sched.items() if k != 'type')}")
    )
    warm = cfg.get("lr_warmup_scheduler") or {}
    if warm:
        rows.append(("warmup", f"{warm.get('type', '-')} {warm.get('warmup_duration', '-')} it"))
    rows.append(
        (
            "amp / ema",
            f"amp={cfg.get('use_amp', False)}, ema={cfg.get('use_ema', False)}"
            + (f" decay={get(cfg, 'ema', 'decay')}" if cfg.get("use_ema") else ""),
        )
    )
    return rows


def metrics_rows(run, rows_named):
    """One table row per named epoch, all evaluator entries in percent."""
    return [[name, row["epoch"], *[fmt_pct(v) for v in run.stats(row)]] for name, row in rows_named]


def loss_components(run):
    """The top-level loss terms (``train_loss_vfl``, not its aux/dn/enc/pre copies), plus other non-loss training stats."""
    keys = list(run.log[-1].keys()) if run.log else []
    main = [k for k in keys if re.fullmatch(r"train_loss_[a-z]+", k)]
    other = [
        k
        for k in keys
        if k.startswith("train_")
        and k not in main
        and not k.startswith("train_loss")
        and k not in ("train_lr", "train_queries")
    ]
    return main, other


def milestone_rows(run, every):
    marks = {r["epoch"] for r in run.log if r["epoch"] % every == 0}
    if run.log:
        marks.add(run.log[-1]["epoch"])
    best = run.best()
    if best:
        marks.add(best["epoch"])
    if run.stage2_start is not None:
        marks.update(e for e in run.epochs if e in (run.stage2_start - 1, run.stage2_start))
    marks.update(e for e, _ in run.console["reloads"])
    rows = []
    for row in run.log:
        e = row["epoch"]
        if e not in marks:
            continue
        note = []
        if best and e == best["epoch"]:
            note.append("best")
        if run.stage2_start is not None and e == run.stage2_start:
            note.append("stage 2")
        if any(e == r for r, _ in run.console["reloads"]):
            note.append("reload")
        rows.append(
            [
                e,
                *[fmt_pct(v) for v in run.stats(row)[:3]],
                fmt_num(row.get("train_loss")),
                fmt_num(row.get("train_lr"), 2),
                fmt_duration(run.console["train_time"].get(e)),
                " ".join(note),
            ]
        )
    return rows


def per_class_section(run, names):
    """Per-class AP from the ``eval/`` dumps: the periodic snapshots and the latest epoch."""
    eval_dir = os.path.join(run.dir, "eval")
    if not os.path.isdir(eval_dir) or not run.log:
        return ""
    snapshots = []
    for name in sorted(os.listdir(eval_dir)):
        m = re.fullmatch(r"(\d+)\.pth", name)
        if m:
            snapshots.append((int(m.group(1)), os.path.join(eval_dir, name)))
    latest_epoch = run.log[-1]["epoch"]
    if os.path.exists(os.path.join(eval_dir, "latest.pth")) and latest_epoch not in {e for e, _ in snapshots}:
        snapshots.append((latest_epoch, os.path.join(eval_dir, "latest.pth")))
    loaded = [(e, load_eval(p)) for e, p in snapshots]
    loaded = [(e, d) for e, d in loaded if d]
    if not loaded:
        return ""
    last_epoch, last = loaded[-1]
    headers = ["class", *[f"AP @{e}" for e, _ in loaded], f"AP50 @{last_epoch}", f"AR @{last_epoch}"]
    rows = []
    for cat in last:
        rows.append(
            [
                names.get(cat, f"id {cat}"),
                *[fmt_pct(d[cat][0]) if cat in d else "-" for _, d in loaded],
                fmt_pct(last[cat][1]),
                fmt_pct(last[cat][2]),
            ]
        )
    return (
        f"\n## Per-class AP\n\nFrom the COCOeval dumps under `eval/` (the epoch-{last_epoch} column is `latest.pth`, "
        f"the run's last evaluated epoch, not necessarily its best). Recall is at the evaluator's largest `maxDets`.\n\n"
        + table(headers, rows)
        + "\n"
    )


def write_report(run, every, per_class, figure_name):
    c = run.console
    out = [f"# {run.name}\n"]

    # headline
    best = run.best()
    if best is None:
        out.append("No evaluated epoch in `log.txt` yet.\n")
    else:
        last = run.log[-1]
        head = [
            ("status", run.status()),
            ("epochs done", f"{len(run.epochs)} of {run.total_epochs} (last logged: {last['epoch']})"),
            (
                "best AP",
                f"**{fmt_pct(run.ap(best))}** at epoch {best['epoch']} (AP50 {fmt_pct(run.stats(best)[1])}, AP75 {fmt_pct(run.stats(best)[2])})",
            ),
            ("last AP", f"{fmt_pct(run.ap(last))} at epoch {last['epoch']}"),
        ]
        best1 = run.best(run.stage_rows(1))
        best2 = run.best(run.stage_rows(2))
        if run.stage2_start is not None and best1 is not None:
            head.append(("best stage 1", f"{fmt_pct(run.ap(best1))} at epoch {best1['epoch']}"))
        if best2 is not None:
            head.append(
                (
                    "best stage 2",
                    f"{fmt_pct(run.ap(best2))} at epoch {best2['epoch']}"
                    + (" (stage 2 never beat stage 1)" if run.ap(best2) <= run.ap(best1) else ""),
                )
            )
        head.append(
            (
                "wall time",
                fmt_duration(run.wall_seconds())
                + (f" for {len(c['train_time'])} epochs" if c["training_time"] is None and c["train_time"] else ""),
            )
        )
        if c["train_time"]:
            head.append(
                (
                    "per epoch",
                    f"train {fmt_duration(run.mean_epoch_seconds() - (sum(c['eval_time'].values()) / len(c['eval_time']) if c['eval_time'] else 0))}, eval {fmt_duration(sum(c['eval_time'].values()) / len(c['eval_time'])) if c['eval_time'] else '-'}; {sum(c['train_s_it'].values()) / len(c['train_s_it']):.3f} s/it over {c['iters_per_epoch']} it",
                )
            )
        if c["max_mem"]:
            head.append(("peak GPU memory", f"{c['max_mem'] / 1024:.1f} GB"))
        if len(c["launches"]) > 1:
            head.append(("launches", f"{len(c['launches'])} (resumed {len(c['launches']) - 1} time(s))"))
        if c["tracebacks"]:
            head.append(("tracebacks", f"{c['tracebacks']} in `console.log`"))
        out.append(table(["", ""], head, ["---", "---"]) + "\n")
        out.append(f"![curves]({figure_name})\n")

    # setup
    out.append("## Setup\n")
    out.append(table(["", ""], setup_rows(run), ["---", "---"]) + "\n")

    if best is not None:
        # metrics of the epochs that matter
        named = [("best", best), ("last", run.log[-1])]
        if run.stage2_start is not None and best1 is not None and best1["epoch"] != best["epoch"]:
            named.insert(1, ("best stage 1", best1))
        out.append("## Metrics\n")
        out.append(f"`{run.metric_key}` of the {run.evaluator}, in percent.\n")
        out.append(table(["epoch", "#", *run.labels], metrics_rows(run, named)) + "\n")

        # queries
        q = run.log[-1].get("test_queries")
        if q:
            out.append(
                f"Queries per validation image at epoch {run.log[-1]['epoch']}: mean {q[0]:.1f}, min {q[1]}, max {q[2]}; "
                f"{100 * q[3]:.1f}% of images had at least as many queries as ground-truth boxes"
                + (f". Training mean {run.log[-1]['train_queries']:.1f}." if "train_queries" in run.log[-1] else ".")
                + "\n"
            )

        # checkpoints
        held = run.checkpoint_epochs()
        files = sorted(f for f in os.listdir(run.dir) if f.endswith(".pth"))
        if files:
            rows = []
            for f in files:
                epoch = held.get(f)
                ap = (
                    next((fmt_pct(run.ap(r)) for r in run.log if r["epoch"] == epoch), "-")
                    if epoch is not None
                    else "-"
                )
                rows.append(
                    [
                        f"`{f}`",
                        epoch if epoch is not None else "?",
                        ap,
                        f"{os.path.getsize(os.path.join(run.dir, f)) / 2**20:.0f} MB",
                    ]
                )
            out.append("## Checkpoints\n")
            out.append(
                "Epochs inferred from `log.txt` and the solver's saving rule (`last.pth` and the numbered copies are stage 1 only).\n"
            )
            out.append(table(["file", "epoch", "AP", "size"], rows) + "\n")

        # stage-2 events
        if c["reloads"] or c["patience"]:
            out.append("## Stage 2 events\n")
            for epoch, decay in c["reloads"]:
                out.append(f"- epoch {epoch}: reloaded `best_stg1.pth`" + (f", EMA decay {decay}" if decay else ""))
            if c["patience"]:
                waits = [n for _, n, _ in c["patience"]]
                out.append(
                    f"- {len(c['patience'])} stage-2 epochs without a new best (longest streak {max(waits)} of patience {c['patience'][0][2]})"
                )
            out.append("")

        # milestones
        out.append("## Progress\n")
        out.append(
            table(
                ["epoch", "AP", "AP50", "AP75", "loss", "lr", "train time", ""],
                milestone_rows(run, every),
                ["---:"] * 7 + ["---"],
            )
            + "\n"
        )

        # losses
        main, other = loss_components(run)
        first, last = run.log[0], run.log[-1]
        rows = [["train_loss", fmt_num(first.get("train_loss")), fmt_num(last.get("train_loss"))]]
        rows += [[k, fmt_num(first.get(k)), fmt_num(last.get(k))] for k in main]
        out.append("## Losses\n")
        out.append(
            f"Epoch means of the top-level terms (the aux, denoising and encoder copies are in `log.txt`), first epoch {first['epoch']} against last epoch {last['epoch']}.\n"
        )
        out.append(table(["term", f"epoch {first['epoch']}", f"epoch {last['epoch']}"], rows) + "\n")
        if other:
            out.append("Other training statistics of the last epoch:\n")
            out.append(table(["stat", "value"], [[k, fmt_num(last[k])] for k in other]) + "\n")

        if per_class:
            out.append(per_class_section(run, category_names(run.cfg)))

    if c["warnings"]:
        out.append("## Warnings\n")
        out += [f"- `{w}`" for w in sorted(c["warnings"])]
        out.append("")

    out.append(f"\n*Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by `tools/analysis/run_report.py`.*\n")
    with open(os.path.join(run.dir, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))


# --------------------------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------------------------


def plot_curves(run, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = run.epochs
    best = run.best()
    c = run.console

    def style(ax, ylabel=None):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("epoch", fontsize=8, color=MUTED)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
        if run.total_epochs:
            ax.set_xlim(0, run.total_epochs - 1)

    def stages(ax):
        marks = set()
        if run.stage2_start is not None:
            marks.add(run.stage2_start)
        marks.update(e for e, _ in c["reloads"])
        for e in sorted(marks):
            ax.axvline(e, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))

    def lines(ax, series, labels, pct=False, log=False):
        scale = 100 if pct else 1
        for i, (values, label) in enumerate(zip(series, labels)):
            xs = [e for e, v in zip(epochs, values) if v is not None]
            ys = [scale * v for v in values if v is not None]
            ax.plot(xs, ys, color=SERIES[i % len(SERIES)], linewidth=1.6, label=label)
        if log:
            ax.set_yscale("log")
        if len(labels) > 1:
            ax.legend(fontsize=8, frameon=False, labelcolor=INK)
        stages(ax)

    panels = []

    def ap_panel(ax):
        lines(ax, [run.metric_series(k) for k in run.labels[:3]], run.labels[:3], pct=True)
        style(ax, "%")
        if best:
            x, y = best["epoch"], 100 * run.ap(best)
            ax.plot([x], [y], "o", color=SERIES[0], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(
                f"best {y:.1f} @ {x}",
                (x, y),
                xytext=(-8, 8),
                textcoords="offset points",
                fontsize=8,
                color=INK,
                ha="right",
            )

    panels.append(("AP", ap_panel))

    size_ap = [k for k in run.labels if k.startswith("AP_")]
    if size_ap:
        panels.append(
            (
                "AP by object size",
                lambda ax: (lines(ax, [run.metric_series(k) for k in size_ap], size_ap, pct=True), style(ax, "%")),
            )
        )
    ar = [k for k in run.labels if k.startswith("AR")]
    if ar:
        panels.append(("AR", lambda ax: (lines(ax, [run.metric_series(k) for k in ar], ar, pct=True), style(ax, "%"))))

    panels.append(("total train loss", lambda ax: (lines(ax, [run.series("train_loss")], ["train_loss"]), style(ax))))
    main, _ = loss_components(run)
    if main:
        short = [k.replace("train_loss_", "") for k in main]
        panels.append(
            (
                "loss terms (top-level)",
                lambda ax: (lines(ax, [run.series(k) for k in main], short, log=True), style(ax)),
            )
        )

    def lr_panel(ax):
        lines(ax, [run.series("train_lr")], ["lr"])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_color(MUTED)
        style(ax)

    panels.append(("learning rate (first param group)", lr_panel))

    if c["train_time"]:

        def time_panel(ax):
            train = [c["train_time"].get(e) for e in epochs]
            evals = [c["eval_time"].get(e) for e in epochs]
            lines(ax, [train, evals], ["train", "eval"])
            style(ax, "seconds")

        panels.append(("epoch time", time_panel))

    counts = {q[0] for r in run.log for q in [r.get("test_queries")] if q} | {
        r["train_queries"] for r in run.log if "train_queries" in r
    }
    if len(counts) > 1:  # dynamic query counts; a fixed count makes an empty panel

        def query_panel(ax):
            test = [r.get("test_queries") for r in run.log]
            train = run.series("train_queries")
            lines(ax, [train, [q[0] if q else None for q in test]], ["train mean", "val mean"])
            xs = [e for e, q in zip(epochs, test) if q]
            ax.fill_between(
                xs,
                [q[1] for q in test if q],
                [q[2] for q in test if q],
                color=SERIES[1],
                alpha=0.15,
                linewidth=0,
                label="val min-max",
            )
            ax.legend(fontsize=8, frameon=False, labelcolor=INK)
            style(ax, "queries / image")

        panels.append(("queries per image", query_panel))

    cols = 3
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.6 * rows), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    axes = list(axes.flat)
    for ax, (title, draw) in zip(axes, panels):
        draw(ax)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    for ax in axes[len(panels) :]:
        ax.axis("off")
    fig.suptitle(run.name, fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "runs", nargs="+", help="output directories of train.py (each gets its own RESULTS.md and curves.png)"
    )
    parser.add_argument("--every", type=int, default=10, help="epoch spacing of the progress table (default 10)")
    parser.add_argument("--no-figure", action="store_true", help="skip curves.png")
    parser.add_argument(
        "--no-per-class", action="store_true", help="skip the per-class table (needs torch to read eval/*.pth)"
    )
    args = parser.parse_args()

    for directory in args.runs:
        if not os.path.exists(os.path.join(directory, "config.yml")):
            print(f"skip {directory}: no config.yml")
            continue
        run = Run(directory)
        print(f"{run.name}: {len(run.epochs)} epochs logged")
        figure = "curves.png"
        if not args.no_figure and run.log:
            plot_curves(run, os.path.join(directory, figure))
        write_report(run, args.every, not args.no_per_class, figure)
        print(
            f"  wrote {os.path.join(run.name, 'RESULTS.md')}"
            + ("" if args.no_figure or not run.log else f" and {figure}")
        )


if __name__ == "__main__":
    main()
