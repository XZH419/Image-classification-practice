from __future__ import annotations

"""Model registry used as a small factory layer.

Model files register themselves through ``@register_model`` when imported.
Training and evaluation scripts can then build models from YAML names without
hard-coding concrete classes in the command-line entry points.
"""

from typing import Callable, Dict

import torch.nn as nn

MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Register model constructor with a decorator."""

    def decorator(cls_or_fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = cls_or_fn
        return cls_or_fn

    return decorator


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        candidates = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: [{candidates}]")
    return MODEL_REGISTRY[name](**kwargs)
