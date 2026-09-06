"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision

from ...core import register

__all__ = ["DomePostProcessor"]


@register()
class DomePostProcessor(nn.Module):
    """
    Decoder outputs -> per-image detections in original-image pixels.

    ``forward`` takes the decoder's ``pred_logits`` [B, Q, C] and ``pred_boxes`` [B, Q, 4]
    (normalized cxcywh) and the (w, h) each image's predictions are to be scaled by, and returns
    one ``{"labels", "boxes", "scores"}`` dict per image, boxes as xyxy pixels.

    With focal (sigmoid) scores every (query, class) pair is a candidate and the top
    ``num_top_queries`` of them are kept, so a query can yield more than one detection; with
    softmax scores each query keeps its best class. The number of queries Q varies per image
    under PAQI, and the cap never exceeds it.

    ``deploy()`` switches the output to the ``(labels, boxes, scores)`` tensors that the ONNX /
    TensorRT export wraps.
    """

    __share__ = ["num_classes", "use_focal_loss", "num_top_queries"]

    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=1500) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        num_queries = logits.shape[1]
        k = min(self.num_top_queries, num_queries)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred = bbox_pred * orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), k, dim=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            scores, index = torch.topk(scores, k, dim=-1)
            labels = labels.gather(dim=1, index=index)
        boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, 4))

        if self.deploy_mode:
            return labels, boxes, scores

        return [dict(labels=lab, boxes=box, scores=sco) for lab, box, sco in zip(labels, boxes, scores)]

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
