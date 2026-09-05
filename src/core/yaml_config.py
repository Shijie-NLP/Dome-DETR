"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ._config import BaseConfig
from .workspace import create
from .yaml_utils import load_config, merge_config, merge_dict

# evaluators that score against the COCO ground truth their validation dataset builds
COCO_STYLE_EVALUATORS = ("VOCEvaluator", "VisDroneEvaluator", "CocoEvaluator")


class YAMLConfig(BaseConfig):
    """
    BaseConfig whose components are built lazily from a yaml file: each property creates its
    object from the registry the first time it is read, using the merged global config.
    """

    def __init__(self, cfg_path: str, **kwargs) -> None:
        super().__init__()

        cfg = load_config(cfg_path)
        cfg = merge_dict(cfg, kwargs)

        self.yaml_cfg = copy.deepcopy(cfg)

        for k in super().__dict__:
            if not k.startswith("_") and k in cfg:
                self.__dict__[k] = cfg[k]

    @property
    def global_cfg(self):
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None and "model" in self.yaml_cfg:
            self._model = create(self.yaml_cfg["model"], self.global_cfg)
        return super().model

    @property
    def postprocessor(self) -> torch.nn.Module:
        if self._postprocessor is None and "postprocessor" in self.yaml_cfg:
            self._postprocessor = create(self.yaml_cfg["postprocessor"], self.global_cfg)
        return super().postprocessor

    @property
    def criterion(self) -> torch.nn.Module:
        if self._criterion is None and "criterion" in self.yaml_cfg:
            self._criterion = create(self.yaml_cfg["criterion"], self.global_cfg)
        return super().criterion

    @property
    def optimizer(self) -> optim.Optimizer:
        if self._optimizer is None and "optimizer" in self.yaml_cfg:
            params = self.get_optim_params(self.yaml_cfg["optimizer"], self.model)
            self._optimizer = create("optimizer", self.global_cfg, params=params)
        return super().optimizer

    @property
    def lr_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        if self._lr_scheduler is None and "lr_scheduler" in self.yaml_cfg:
            self._lr_scheduler = create("lr_scheduler", self.global_cfg, optimizer=self.optimizer)
            print(f"Initial lr: {self._lr_scheduler.get_last_lr()}")
        return super().lr_scheduler

    @property
    def lr_warmup_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        if self._lr_warmup_scheduler is None and "lr_warmup_scheduler" in self.yaml_cfg:
            self._lr_warmup_scheduler = create("lr_warmup_scheduler", self.global_cfg, lr_scheduler=self.lr_scheduler)
        return super().lr_warmup_scheduler

    @property
    def train_dataloader(self) -> DataLoader:
        if self._train_dataloader is None and "train_dataloader" in self.yaml_cfg:
            self._train_dataloader = self.build_dataloader("train_dataloader")
        return super().train_dataloader

    @property
    def val_dataloader(self) -> DataLoader:
        if self._val_dataloader is None and "val_dataloader" in self.yaml_cfg:
            self._val_dataloader = self.build_dataloader("val_dataloader")
        return super().val_dataloader

    @property
    def ema(self) -> torch.nn.Module:
        if self._ema is None and self.yaml_cfg.get("use_ema", False):
            self._ema = create("ema", self.global_cfg, model=self.model)
        return super().ema

    @property
    def scaler(self):
        if self._scaler is None and self.yaml_cfg.get("use_amp", False):
            self._scaler = create("scaler", self.global_cfg)
        return super().scaler

    @property
    def evaluator(self):
        if self._evaluator is None and "evaluator" in self.yaml_cfg:
            evaluator_type = self.yaml_cfg["evaluator"]["type"]
            if evaluator_type not in COCO_STYLE_EVALUATORS:
                raise NotImplementedError(f"evaluator {evaluator_type!r}; known: {COCO_STYLE_EVALUATORS}")
            from ..data import get_coco_api_from_dataset

            coco_gt = get_coco_api_from_dataset(self.val_dataloader.dataset)
            self._evaluator = create("evaluator", self.global_cfg, coco_gt=coco_gt)
        return super().evaluator

    @staticmethod
    def get_optim_params(cfg: dict, model: nn.Module):
        """
        Parameter groups from the optimizer config: each entry of ``params`` names a regex over
        parameter names, and whatever no entry matched goes into a final default group.

        E.g.:
            ^(?=.*a)(?=.*b).*$  means including a and b
            ^(?=.*(?:a|b)).*$   means including a or b
            ^(?=.*a)(?!.*b).*$  means including a, but not b
        """
        assert "type" in cfg, "optimizer config needs a `type`"
        cfg = copy.deepcopy(cfg)

        if "params" not in cfg:
            return model.parameters()

        assert isinstance(cfg["params"], list), "optimizer `params` must be a list of groups"

        param_groups = []
        visited = []
        for pg in cfg["params"]:
            pattern = pg["params"]
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and re.findall(pattern, k)}
            pg["params"] = params.values()
            param_groups.append(pg)
            visited.extend(params.keys())

        names = [k for k, v in model.named_parameters() if v.requires_grad]

        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            param_groups.append({"params": params.values()})
            visited.extend(params.keys())

        assert len(visited) == len(names), "every trainable parameter must land in exactly one group"

        return param_groups

    @staticmethod
    def get_rank_batch_size(cfg):
        """The per-rank batch size: ``batch_size`` as given, or ``total_batch_size`` split over the ranks."""
        assert ("total_batch_size" in cfg) != ("batch_size" in cfg), (
            "give exactly one of `batch_size` and `total_batch_size`"
        )

        total_batch_size = cfg.get("total_batch_size", None)
        if total_batch_size is None:
            return cfg["batch_size"]

        from ..misc import dist_utils

        world_size = dist_utils.get_world_size()
        assert total_batch_size % world_size == 0, "total_batch_size should be divisible by world size"
        return total_batch_size // world_size

    def build_dataloader(self, name: str):
        bs = self.get_rank_batch_size(self.yaml_cfg[name])
        global_cfg = self.global_cfg
        # total_batch_size is ours, not DataLoader's
        global_cfg[name].pop("total_batch_size", None)
        print(f"building {name} with batch_size={bs}...")
        loader = create(name, global_cfg, batch_size=bs)
        loader.shuffle = self.yaml_cfg[name].get("shuffle", False)
        return loader
