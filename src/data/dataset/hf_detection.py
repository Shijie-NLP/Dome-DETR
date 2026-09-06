"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

Detection datasets read straight from the Hugging Face Hub.

A dataset here is a Hub repo plus a rule for turning one row's ``objects`` into boxes. The base
class owns everything that does not depend on the dataset: downloading and caching the parquet,
decoding one row into the ``(image, target)`` pair the transforms and the model expect, and
building the in-memory COCO ground truth the evaluators score against. A subclass only states
which repo it reads and how its annotation columns map to ``(boxes, labels, iscrowd)``.

The rows on the Hub keep the official annotations whole, so the dataset-specific policy -- which
pseudo-classes to drop, what counts as an ignore region, how a 1-indexed VOC box becomes a
0-indexed one -- lives in the subclass's ``parse_objects`` and nowhere else. The same function
feeds both training targets and the evaluation ground truth, so the two can never disagree.
"""

from collections.abc import Callable

import torch
import torch.utils.data
from faster_coco_eval import COCO
from PIL import Image

from .._misc import convert_to_tv_tensor
from ._dataset import DetDataset

__all__ = ["HFDetection", "get_coco_api_from_dataset"]


class HFDetection(DetDataset):
    """
    A detection split of a Hub dataset repo.

    Rows are expected to carry ``id`` (string), ``image``, ``width``, ``height`` and ``objects``
    (a dict of parallel lists). Subclasses set ``REPO``, ``CONFIG`` and ``CATEGORIES`` and
    implement ``parse_objects``.
    """

    __inject__ = ["transforms"]

    REPO: str = None
    CONFIG: str | None = None
    # (category id, name) pairs of the classes that are scored. Ids are what ``parse_objects``
    # emits as labels, so they are also what the model predicts.
    CATEGORIES: list[tuple[int, str]] = []

    def __init__(
        self,
        split: str,
        repo: str | None = None,
        config: str | None = None,
        transforms: Callable | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
    ):
        self.repo = repo or self.REPO
        self.config = config or self.CONFIG
        self.split = split
        self.revision = revision
        self.hf = self._load_split(self.repo, self.config, split, revision, cache_dir)
        # every column but the pixels: what the COCO ground truth is built from, without decoding
        # a single image
        self.hf_meta = self.hf.remove_columns("image")
        self.transforms = transforms
        self._coco = None

    @staticmethod
    def _load_split(repo, config, split, revision, cache_dir):
        """
        The rows of ``split`` (a name, or names joined with ``+``), downloading only the parquet
        files of the splits it names. ``load_dataset(repo, split=...)`` would fetch every split
        of the config first and select afterwards, which for a 27 GB dataset is the difference
        between a test set and the whole thing. A repo that is not parquet-backed falls back to
        the plain call.
        """
        import re

        from datasets import load_dataset, load_dataset_builder

        builder = load_dataset_builder(repo, config, revision=revision, cache_dir=cache_dir)
        data_files = builder.config.data_files  # split name -> resolved file paths (revision pinned)
        if not data_files:
            return load_dataset(repo, config, split=split, revision=revision, cache_dir=cache_dir)

        names = [re.match(r"[^\[\s]+", part.strip()).group(0) for part in split.split("+")]
        unknown = [name for name in names if name not in data_files]
        if unknown:
            raise ValueError(f"{repo} has no split {unknown}; available: {list(data_files)}")
        files = {name: [str(f) for f in data_files[name]] for name in dict.fromkeys(names)}
        return load_dataset("parquet", data_files=files, split=split, cache_dir=cache_dir)

    def parse_objects(self, objects: dict[str, list]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One row's ``objects`` as ``(boxes, labels, iscrowd)``: boxes ``[N, 4]`` float xyxy in
        pixels of the original image, labels ``[N]`` int64 category ids, iscrowd ``[N]`` int64
        with 1 for boxes that are ignore regions rather than objects. Ignore regions are left out
        of the training target and kept in the evaluation ground truth, where the evaluator
        treats them as COCO crowd boxes: a detection on one is neither a hit nor a false positive.
        """
        raise NotImplementedError

    def file_name(self, row_id: str) -> str:
        """The image file name recorded in the COCO ground truth, for tools that want one."""
        return row_id + ".jpg"

    def __len__(self):
        return len(self.hf)

    def load_item(self, index: int):
        row = self.hf[index]
        image: Image.Image = row["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size
        boxes, labels, iscrowd = self.parse_objects(row["objects"])
        return image, self._make_target(index, boxes, labels, iscrowd, w, h)

    def _make_target(self, index, boxes, labels, iscrowd, w, h):
        # the training target: real objects only, clamped to the image, degenerate boxes dropped
        boxes = boxes.clone()
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        keep = (iscrowd == 0) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, labels, iscrowd = boxes[keep], labels[keep], iscrowd[keep]

        target = {}
        target["boxes"] = convert_to_tv_tensor(boxes, key="boxes", box_format="xyxy", spatial_size=[h, w])
        target["labels"] = labels
        target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([index])
        target["idx"] = torch.tensor([index])
        target["orig_size"] = torch.tensor([w, h])
        return target

    @property
    def coco(self) -> COCO:
        """
        The split as a COCO ground truth, built once from the annotation columns. Image ids are
        row indices, which is also what ``load_item`` writes into ``target['image_id']``, so a
        prediction keyed by that id lands on the right image.
        """
        if self._coco is None:
            self._coco = self.build_coco()
        return self._coco

    def build_coco(self) -> COCO:
        images, annotations = [], []
        ann_id = 1  # annotation ids start at 1: id 0 is falsy and trips pycocotools
        for index, row in enumerate(self.hf_meta):
            w, h = row["width"], row["height"]
            images.append({"id": index, "file_name": self.file_name(row["id"]), "width": w, "height": h})
            boxes, labels, iscrowd = self.parse_objects(row["objects"])
            for (x1, y1, x2, y2), label, crowd in zip(boxes.tolist(), labels.tolist(), iscrowd.tolist()):
                bw, bh = x2 - x1, y2 - y1
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": index,
                        "category_id": label,
                        "bbox": [x1, y1, bw, bh],
                        "area": bw * bh,
                        "iscrowd": crowd,
                        "ignore": crowd,
                    }
                )
                ann_id += 1

        coco = COCO()
        coco.dataset = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": i, "name": name} for i, name in self.CATEGORIES],
        }
        coco.createIndex()
        return coco

    @property
    def category2name(self):
        return dict(self.CATEGORIES)

    def __getstate__(self):
        # dataloader workers get the rows, not the ground truth: it is only read in the main
        # process, and pickling a few hundred thousand annotations per worker is pure cost
        state = self.__dict__.copy()
        state["_coco"] = None
        return state

    def __repr__(self):
        s = f"{self.__class__.__name__}(repo={self.repo!r}, config={self.config!r}, split={self.split!r}, len={len(self)})"
        if self.transforms is not None:
            s += f"\n  transforms:\n    {self.transforms!r}"
        return s


def get_coco_api_from_dataset(dataset) -> COCO:
    """The COCO ground truth of a dataset, looking through any Subset wrappers."""
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if not hasattr(dataset, "coco"):
        raise TypeError(f"{type(dataset).__name__} has no COCO ground truth; expected an HFDetection dataset")
    return dataset.coco
