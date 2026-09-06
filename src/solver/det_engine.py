"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import math
import sys
from collections.abc import Iterable

import torch
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data.dataset.coco_eval import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..misc.visualizer import SAVE_TEST_VISUALIZE_RESULT, PredictionDumper, dump_training_targets
from ..optim import ModelEMA, Warmup


def to_device(targets: list[dict], device) -> list[dict]:
    """The per-image target dicts with every tensor moved to ``device``."""
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


def optimizer_step(loss: torch.Tensor, model, optimizer, scaler: GradScaler | None, max_norm: float) -> None:
    """Backward, optional gradient clipping and the optimizer step, through the GradScaler when there is one."""
    optimizer.zero_grad()
    if scaler is None:
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        return

    scaler.scale(loss).backward()
    if max_norm > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    scaler.step(optimizer)
    scaler.update()


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    *,
    print_freq: int = 10,
    writer: SummaryWriter | None = None,
    ema: ModelEMA | None = None,
    scaler: GradScaler | None = None,
    lr_warmup_scheduler: Warmup | None = None,
):
    """One epoch over ``data_loader``; mixed precision when a ``scaler`` is given. Returns the averaged meters."""
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    use_amp = scaler is not None
    device_type = torch.device(device).type

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = to_device(targets, device)
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        dump_training_targets(samples, targets)

        with torch.autocast(device_type=device_type, enabled=use_amp, cache_enabled=True):
            outputs = model(samples, targets=targets)

        if not torch.isfinite(outputs["pred_boxes"]).all():
            # keep the weights that produced the non-finite boxes, for a post-mortem
            print(outputs["pred_boxes"])
            state = dist_utils.remove_module_prefix(model.state_dict())
            dist_utils.save_on_master({"model": state}, "./NaN.pth")

        # the loss is always computed in full precision
        loss_dict = criterion(outputs, targets, **metas)
        loss: torch.Tensor = sum(loss_dict.values())
        optimizer_step(loss, model, optimizer, scaler, max_norm)

        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def prediction_scale(target):
    """
    The (w, h) that maps a prediction normalized to the network input back to the original image.
    That is ``orig_size``, unless the collate function padded the image: then the input is larger
    than the resized image by ``padded_size / resized_size``, and so must be the scale.
    """
    if "padded_size" in target:
        return target["orig_size"] * target["padded_size"] / target["resized_size"]
    return target["orig_size"]


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    # DeFE statistics: how often the predicted query budget covers the ground truth, and its mean
    use_defe = dist_utils.de_parallel(model).encoder.use_defe
    total_defe_samples = 0
    ample_defe_predictions = 0
    total_anchor_num = 0

    with PredictionDumper(SAVE_TEST_VISUALIZE_RESULT) as dumper:
        for samples, targets in metric_logger.log_every(data_loader, 10, header):
            samples = samples.to(device)
            targets = to_device(targets, device)

            outputs = model(samples, targets=targets)
            orig_target_sizes = torch.stack([prediction_scale(t) for t in targets], dim=0)
            results = postprocessor(outputs, orig_target_sizes)

            if dumper.enabled:
                coco = data_loader.dataset.coco
                file_names = [coco.loadImgs(t["image_id"].item())[0]["file_name"] for t in targets]
                scale_factor = float(samples[0].shape[1] / orig_target_sizes[0][0])
                dumper.submit(samples, targets, results, file_names, scale_factor)

            res = {target["image_id"].item(): output for target, output in zip(targets, results)}
            coco_evaluator.update(res)

            if use_defe:
                pred_defe = outputs["batch_queries_num"][0]
                if pred_defe >= targets[0]["labels"].shape[0]:
                    ample_defe_predictions += 1
                total_defe_samples += 1
                total_anchor_num += pred_defe

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if use_defe and total_defe_samples:
        print("defe Ample Rate:", ample_defe_predictions / total_defe_samples)
        print("defe Average Anchor Number:", total_anchor_num / total_defe_samples)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = {}
    if "bbox" in coco_evaluator.iou_types:
        stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
    if "segm" in coco_evaluator.iou_types:
        stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    return stats, coco_evaluator
