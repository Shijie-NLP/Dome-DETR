"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.

torch's optimizers, lr schedulers and GradScaler, registered so that a yaml can name them
(``optimizer: {type: AdamW, ...}``). Nothing here is subclassed; the registered names are the
torch classes themselves.
"""

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.amp import GradScaler as _GradScaler

from ..core import register

__all__ = ["SGD", "Adam", "AdamW", "CosineAnnealingLR", "GradScaler", "LambdaLR", "MultiStepLR", "OneCycleLR"]

SGD = register()(optim.SGD)
Adam = register()(optim.Adam)
AdamW = register()(optim.AdamW)

MultiStepLR = register()(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register()(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register()(lr_scheduler.OneCycleLR)
LambdaLR = register()(lr_scheduler.LambdaLR)

# torch.amp.GradScaler defaults to device="cuda", which is what torch.cuda.amp.GradScaler was
GradScaler = register()(_GradScaler)
