from __future__ import annotations

"""Reproducibility helper for random split and model initialization control."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Ensure strong reproducibility across dataloading and training."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
