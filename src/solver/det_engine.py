"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import concurrent.futures
import math
import os
import sys
from collections.abc import Iterable

import torch
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from tools.concatenate_images import concatenate_images
from tools.visualize_image_annotation import visualize_detection

from ..data.dataset.coco_eval import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..optim import ModelEMA, Warmup

# set to "True" to dump every training sample with its boxes (train) or every prediction next to
# its ground truth (test) as images, for eyeballing the pipeline
SAVE_INTERMEDIATE_VISUALIZE_RESULT = os.environ.get("SAVE_INTERMEDIATE_VISUALIZE_RESULT", "False") == "True"
SAVE_TEST_VISUALIZE_RESULT = os.environ.get("SAVE_TEST_VISUALIZE_RESULT", "False") == "True"


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    **kwargs,
):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer")

    ema: ModelEMA = kwargs.get("ema")
    scaler: GradScaler = kwargs.get("scaler")
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler")

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if SAVE_INTERMEDIATE_VISUALIZE_RESULT:
            for b, target in enumerate(targets):
                image = samples[b].cpu()
                _, H, W = image.shape
                target_cpu = {}
                for k, v in target.items():
                    if k == "boxes":
                        target_cpu[k] = v.cpu().detach().clone() * torch.tensor([W, H, W, H])
                    else:
                        target_cpu[k] = v.cpu().detach().clone()
                visualize_detection(image, target_cpu, "sample_gt", return_image=False, type="xywh")

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if not torch.isfinite(outputs["pred_boxes"]).all():
                # keep the weights that produced the non-finite boxes, for a post-mortem
                print(outputs["pred_boxes"])
                state = {k.replace("module.", ""): v for k, v in model.state_dict().items()}
                dist_utils.save_on_master({"model": state}, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

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


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
):
    if SAVE_TEST_VISUALIZE_RESULT:
        os.makedirs("visualize_all", exist_ok=True)
        print("Saving visualize results to visualize_all/")
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

    max_pending_tasks = 256
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        pending_futures = set()

        for samples, targets in metric_logger.log_every(data_loader, 10, header):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples, targets=targets)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_target_sizes)

            if SAVE_TEST_VISUALIZE_RESULT:
                coco = data_loader.dataset.coco
                file_names = [coco.loadImgs(t["image_id"].item())[0]["file_name"] for t in targets]
                scale_factor = float(samples[0].shape[1] / orig_target_sizes[0][0])

                # bound the backlog so the thread pool cannot outgrow memory
                while len(pending_futures) >= max_pending_tasks:
                    _, pending_futures = concurrent.futures.wait(
                        pending_futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                for i in range(len(targets)):
                    args = (
                        samples[i].cpu(),
                        {k: v.cpu() for k, v in targets[i].items()},
                        {k: v.cpu() for k, v in results[i].items()},
                        file_names[i],
                        scale_factor,
                    )
                    pending_futures.add(executor.submit(process_image_pair, args))

            res = {target["image_id"].item(): output for target, output in zip(targets, results)}
            coco_evaluator.update(res)

            if use_defe:
                pred_defe = outputs["batch_queries_num"][0]
                if pred_defe >= targets[0]["labels"].shape[0]:
                    ample_defe_predictions += 1
                total_defe_samples += 1
                total_anchor_num += pred_defe

        concurrent.futures.wait(pending_futures)

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


def process_image_pair(args):
    sample, target, result, filename, scale_factor = args
    sample_img = visualize_detection(sample, target, f"sample_{filename}", return_image=True)
    result_img = visualize_detection(sample, result, f"result_{filename}", scale_factor=scale_factor, return_image=True)
    concatenate_images(sample_img, result_img, output_path=f"visualize_all/{filename}")
