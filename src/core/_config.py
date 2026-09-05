"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

__all__ = ["BaseConfig"]


class BaseConfig:
    """
    Everything a solver reads: the runtime settings as plain attributes, and the training
    components (model, criterion, loaders, optimizer, ...) as properties. Here the components are
    just held; ``YAMLConfig`` overrides the properties to build each one from the yaml the first
    time it is asked for. Public attribute names double as the yaml keys ``YAMLConfig`` copies in.
    """

    def __init__(self) -> None:
        super().__init__()

        self.task: str = None

        # components, built lazily by the subclass or set from outside
        self._model: nn.Module = None
        self._postprocessor: nn.Module = None
        self._criterion: nn.Module = None
        self._optimizer: Optimizer = None
        self._lr_scheduler: LRScheduler = None
        self._lr_warmup_scheduler: LRScheduler = None
        self._train_dataloader: DataLoader = None
        self._val_dataloader: DataLoader = None
        self._ema: nn.Module = None
        self._scaler: GradScaler = None
        self._evaluator: Any = None
        self._writer: SummaryWriter = None

        # runtime
        self.resume: str = None
        self.tuning: str = None

        self.epoches: int = None
        self.last_epoch: int = -1

        self.use_amp: bool = False
        self.use_ema: bool = False
        self.ema_decay: float = 0.9999
        self.ema_warmups: int = 2000
        self.sync_bn: bool = False
        self.clip_max_norm: float = 0.0
        self.find_unused_parameters: bool = None

        self.seed: int = None
        self.print_freq: int = None
        self.checkpoint_freq: int = 1
        self.output_dir: str = None
        self.summary_dir: str = None
        self.device: str = ""

    @property
    def model(self) -> nn.Module:
        return self._model

    @model.setter
    def model(self, m):
        assert isinstance(m, nn.Module), f"{type(m)} != nn.Module, please check your model class"
        self._model = m

    @property
    def postprocessor(self) -> nn.Module:
        return self._postprocessor

    @postprocessor.setter
    def postprocessor(self, m):
        assert isinstance(m, nn.Module), f"{type(m)} != nn.Module, please check your postprocessor class"
        self._postprocessor = m

    @property
    def criterion(self) -> nn.Module:
        return self._criterion

    @criterion.setter
    def criterion(self, m):
        assert isinstance(m, nn.Module), f"{type(m)} != nn.Module, please check your criterion class"
        self._criterion = m

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, m):
        assert isinstance(m, Optimizer), f"{type(m)} != optim.Optimizer, please check your optimizer class"
        self._optimizer = m

    @property
    def lr_scheduler(self) -> LRScheduler:
        return self._lr_scheduler

    @lr_scheduler.setter
    def lr_scheduler(self, m):
        assert isinstance(m, LRScheduler), f"{type(m)} != LRScheduler, please check your scheduler class"
        self._lr_scheduler = m

    @property
    def lr_warmup_scheduler(self) -> LRScheduler:
        return self._lr_warmup_scheduler

    @lr_warmup_scheduler.setter
    def lr_warmup_scheduler(self, m):
        self._lr_warmup_scheduler = m

    @property
    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    @train_dataloader.setter
    def train_dataloader(self, loader):
        self._train_dataloader = loader

    @property
    def val_dataloader(self) -> DataLoader:
        return self._val_dataloader

    @val_dataloader.setter
    def val_dataloader(self, loader):
        self._val_dataloader = loader

    @property
    def ema(self) -> nn.Module:
        if self._ema is None and self.use_ema and self.model is not None:
            from ..optim import ModelEMA

            self._ema = ModelEMA(self.model, self.ema_decay, self.ema_warmups)
        return self._ema

    @ema.setter
    def ema(self, obj):
        self._ema = obj

    @property
    def scaler(self) -> GradScaler:
        if self._scaler is None and self.use_amp and torch.cuda.is_available():
            self._scaler = GradScaler("cuda")
        return self._scaler

    @scaler.setter
    def scaler(self, obj: GradScaler):
        self._scaler = obj

    @property
    def evaluator(self):
        return self._evaluator

    @evaluator.setter
    def evaluator(self, fn):
        assert isinstance(fn, Callable), f"{type(fn)} must be Callable"
        self._evaluator = fn

    @property
    def writer(self) -> SummaryWriter:
        if self._writer is None:
            if self.summary_dir:
                self._writer = SummaryWriter(self.summary_dir)
            elif self.output_dir:
                self._writer = SummaryWriter(Path(self.output_dir) / "summary")
        return self._writer

    @writer.setter
    def writer(self, m):
        assert isinstance(m, SummaryWriter), f"{type(m)} must be SummaryWriter"
        self._writer = m

    def __repr__(self):
        return "".join(f"{k}: {v}\n" for k, v in self.__dict__.items() if not k.startswith("_"))
