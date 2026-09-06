"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

Debug visualisation, all of it opt-in through two environment variables so the training loop
and the model stay free of plotting code:

- ``SAVE_INTERMEDIATE_VISUALIZE_RESULT=True`` dumps every training sample with its boxes and the
  intermediate feature / density maps the model hands to ``dump_feature_map`` under ``visualize/``;
- ``SAVE_TEST_VISUALIZE_RESULT=True`` writes, for every validation image, the ground truth and
  the predictions side by side under ``visualize_all/``.

matplotlib is imported lazily so that nothing here costs anything when the flags are off.
"""

import concurrent.futures
import io
import os

import numpy as np
import PIL
import torch
from PIL import Image

__all__ = [
    "SAVE_INTERMEDIATE_VISUALIZE_RESULT",
    "SAVE_TEST_VISUALIZE_RESULT",
    "PredictionDumper",
    "concatenate_images",
    "dump_boxes",
    "dump_feature_map",
    "dump_training_targets",
    "show_sample",
    "visualize_detection",
    "visualize_src_flatten",
]

SAVE_INTERMEDIATE_VISUALIZE_RESULT = os.environ.get("SAVE_INTERMEDIATE_VISUALIZE_RESULT", "False") == "True"
SAVE_TEST_VISUALIZE_RESULT = os.environ.get("SAVE_TEST_VISUALIZE_RESULT", "False") == "True"

INTERMEDIATE_DIR = "visualize"
TEST_DIR = "visualize_all"


def show_sample(sample):
    """Show one ``(image, target)`` of a detection dataset with its boxes, interactively."""
    import matplotlib.pyplot as plt
    from torchvision.transforms.v2 import functional as F  # noqa: N812
    from torchvision.utils import draw_bounding_boxes

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image(image)

    image = F.to_dtype(image, torch.uint8, scale=True)
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------------------------
# intermediate feature maps


def visualize_src_flatten(src_flatten, spatial_shapes, savename="feature", is_flatten=True):
    """
    Save the channel sum of one or more feature maps as heatmaps under ``visualize/``.

    ``is_flatten`` set: ``src_flatten`` is ``[B, sum(h*w), C]`` and ``spatial_shapes`` lists the
    ``(h, w)`` of each level in order. Otherwise ``src_flatten`` is a list of ``[B, h, w, C]`` maps.
    """
    import matplotlib.pyplot as plt

    if is_flatten:
        bs, _, c = src_flatten.shape
        visual_features = []
        start_idx = 0
        for h, w in spatial_shapes:
            end_idx = start_idx + h * w
            visual_features.append(src_flatten[:, start_idx:end_idx, :].view(bs, h, w, c))
            start_idx = end_idx
    else:
        visual_features = src_flatten

    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    for lvl, feat in enumerate(visual_features):
        aggregated = feat.sum(dim=-1)  # [B, h, w]
        aggregated = (aggregated - aggregated.min()) / (aggregated.max() - aggregated.min())
        plt.figure()
        plt.title(f"Level {lvl} - Channel Sum")
        plt.imshow(aggregated.squeeze(0).cpu().detach().numpy(), cmap="plasma")
        plt.colorbar()
        plt.savefig(f"{INTERMEDIATE_DIR}/{savename}_level_{lvl}_channel_sum.png")
        plt.close()


def dump_feature_map(name: str, feature: torch.Tensor) -> None:
    """Save a ``[B, C, H, W]`` map under ``name`` when SAVE_INTERMEDIATE_VISUALIZE_RESULT is set; else nothing."""
    if SAVE_INTERMEDIATE_VISUALIZE_RESULT:
        visualize_src_flatten(feature.permute(0, 2, 3, 1), [tuple(feature.shape[2:4])], name, is_flatten=False)


# ----------------------------------------------------------------------------------------------
# images with boxes


def visualize_detection(
    samples,
    targets,
    savename="output",
    threshold=0.5,
    scale_factor=1.0,
    return_image=False,
    point_mode=False,
    show_label=True,
    area_filter=None,
    type="xyxy",  # noqa: A002
):
    """
    Draw ``targets['boxes']`` (and labels / scores when present) on the first image of
    ``samples``. Boxes are ``type`` = ``xyxy`` or ``xywh`` (centre + size), scaled by
    ``scale_factor``; predictions below ``threshold`` and, with ``area_filter``, boxes at least
    that large are left out. ``point_mode`` draws box centres instead of rectangles. Returns a
    PIL image when ``return_image`` is set, else saves ``visualize/<savename>_rgb.png``.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    img_tensor = samples[0].cpu().detach() if samples.dim() == 4 else samples.cpu().detach()
    img = img_tensor.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    fig = Figure(figsize=(12, 12))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.imshow(img)

    targets = targets[0] if isinstance(targets, list) else targets
    boxes = targets["boxes"].cpu().numpy() * scale_factor
    labels = targets["labels"].cpu().numpy() if "labels" in targets else np.array(["unknown"] * len(boxes))
    scores = targets["scores"].cpu().numpy() if "scores" in targets else None

    if area_filter is not None:
        if type == "xyxy":
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            areas = boxes[:, 2] * boxes[:, 3]
        keep = areas < area_filter
        boxes, labels = boxes[keep], labels[keep]
        scores = scores[keep] if scores is not None else None

    if scores is not None:
        keep = scores > threshold
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    if not point_mode:
        for i, (box, label) in enumerate(zip(boxes, labels)):
            if type == "xywh":
                cx, cy, w, h = box
                x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            else:
                x1, y1, x2, y2 = box
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="lime", facecolor="none"))
            if show_label:
                text = f"Class {label}" + (f" {float(scores[i]):.3f}" if scores is not None else "")
                ax.text(
                    x1,
                    y1 - 5,
                    text,
                    color="lime",
                    fontsize=10,
                    bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"),
                )
    else:
        for box in boxes:
            if type == "xywh":
                x, y = box[0], box[1]
            else:
                x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            ax.scatter(x, y, s=3, c="lime", marker="o")

    ax.axis("off")

    if return_image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        buf.seek(0)
        with Image.open(buf) as pil_image:
            return pil_image.copy()

    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    fig.savefig(f"{INTERMEDIATE_DIR}/{savename}_rgb.png", bbox_inches="tight", dpi=300)
    return None


def concatenate_images(img1, img2, output_path=None, background=(255, 255, 255)):
    """Two PIL images side by side, top-aligned, the shorter one padded with ``background``."""
    w1, h1 = img1.size
    w2, h2 = img2.size
    height = max(h1, h2)
    canvas = Image.new("RGB", (w1 + w2, height), background)
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (w1, 0))
    if output_path:
        canvas.save(output_path)
    return canvas


def dump_training_targets(samples: torch.Tensor, targets: list[dict]) -> None:
    """
    Save every image of a training batch with its ground-truth boxes, when
    SAVE_INTERMEDIATE_VISUALIZE_RESULT is set. Training boxes are normalized cxcywh; they are
    scaled back to pixels here.
    """
    if not SAVE_INTERMEDIATE_VISUALIZE_RESULT:
        return
    for image, target in zip(samples, targets):
        image = image.cpu()
        _, h, w = image.shape
        target_cpu = {k: v.cpu().detach().clone() for k, v in target.items()}
        target_cpu["boxes"] = target_cpu["boxes"] * torch.tensor([w, h, w, h])
        visualize_detection(image, target_cpu, "sample_gt", return_image=False, type="xywh")


def dump_boxes(name, image, boxes, labels=None, scores=None) -> None:
    """
    Save the first image of ``image`` twice under ``visualize/``: with ``boxes`` (normalized
    cxcywh, ``[N, 4]``) drawn as centre points (``<name>_point``) and as rectangles (``<name>``).
    Predictions can pass ``labels`` and ``scores``; scores below 0.5 are left out.
    """
    _, h, w = image[0].shape
    target = {"boxes": boxes.detach() * boxes.new_tensor([w, h, w, h])}
    if labels is not None:
        target["labels"] = labels.detach()
    if scores is not None:
        target["scores"] = scores.detach()
    visualize_detection(image, target, f"{name}_point", point_mode=True, type="xywh")
    visualize_detection(image, target, name, show_label=False, type="xywh")


def save_prediction_pair(sample, target, result, filename, scale_factor):
    """One validation image: ground truth on the left, predictions on the right, saved under ``visualize_all/``."""
    sample_img = visualize_detection(sample, target, f"sample_{filename}", return_image=True)
    result_img = visualize_detection(sample, result, f"result_{filename}", scale_factor=scale_factor, return_image=True)
    concatenate_images(sample_img, result_img, output_path=f"{TEST_DIR}/{filename}")


class PredictionDumper:
    """
    Writes ground-truth / prediction pairs for a validation pass on a thread pool, with a bounded
    backlog so the pool cannot outgrow memory. Does nothing unless ``enabled``.

        with PredictionDumper(SAVE_TEST_VISUALIZE_RESULT) as dumper:
            for ...:
                dumper.submit(samples, targets, results, file_names, scale_factor)
    """

    def __init__(self, enabled: bool, max_workers: int = 32, max_pending: int = 256):
        self.enabled = enabled
        self.max_pending = max_pending
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) if enabled else None
        self._pending = set()
        if enabled:
            os.makedirs(TEST_DIR, exist_ok=True)
            print(f"Saving visualize results to {TEST_DIR}/")

    def submit(self, samples, targets, results, file_names, scale_factor):
        if not self.enabled:
            return
        while len(self._pending) >= self.max_pending:
            _, self._pending = concurrent.futures.wait(self._pending, return_when=concurrent.futures.FIRST_COMPLETED)
        for i, filename in enumerate(file_names):
            args = (
                samples[i].cpu(),
                {k: v.cpu() for k, v in targets[i].items()},
                {k: v.cpu() for k, v in results[i].items()},
                filename,
                scale_factor,
            )
            self._pending.add(self._executor.submit(save_prediction_pair, *args))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._executor is not None:
            concurrent.futures.wait(self._pending)
            self._executor.shutdown()
