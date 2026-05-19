from __future__ import annotations

"""Command line entry point for training STL-10 image classifiers.

The script reads a YAML experiment config, fixes all random seeds, builds the
registered model, splits only the training folder into train/validation sets,
and saves the best validation checkpoint plus learning-curve artifacts.
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml

from engine.trainer import BaseTrainer
from models import base_cnn, resnet_lite  # noqa: F401  # trigger model registration
from models.registry import build_model
from utils.data_loader import build_train_valid_loaders
from utils.seed import seed_everything


@dataclass
class DataConfig:
    train_dir: str = "./data/train"
    test_dir: str = "./data/test"
    image_size: int = 96
    train_size: int = 5600
    valid_size: int = 1400
    num_workers: int = 4
    pin_memory: bool = True
    aug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = "BaseCNN"
    params: Dict[str, Any] = field(default_factory=lambda: {"num_classes": 10})


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    epochs: int = 30
    use_amp: bool = True
    optimizer: str = "sgd"
    scheduler: Any = None


@dataclass
class SaveConfig:
    output_dir: str = "./results"
    best_model_name: str = "best_base_model.pth"
    loss_curve_name: str = "loss_curve.png"
    acc_curve_name: str = "accuracy_curve.png"


@dataclass
class AppConfig:
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    save: SaveConfig = field(default_factory=SaveConfig)


def load_config(config_path: str) -> AppConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    return AppConfig(
        seed=raw_cfg.get("seed", 42),
        device=raw_cfg.get("device", "auto"),
        data=DataConfig(**raw_cfg.get("data", {})),
        model=ModelConfig(**raw_cfg.get("model", {})),
        train=TrainConfig(**raw_cfg.get("train", {})),
        save=SaveConfig(**raw_cfg.get("save", {})),
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    optimizer_name = cfg.optimizer.lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train STL-10 classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)

    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    use_aug = bool(cfg.data.aug)
    train_loader, valid_loader, idx_to_class = build_train_valid_loaders(
        train_dir=cfg.data.train_dir,
        batch_size=cfg.train.batch_size,
        image_size=cfg.data.image_size,
        train_size=cfg.data.train_size,
        valid_size=cfg.data.valid_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        seed=cfg.seed,
        use_aug=use_aug,
    )
    print(f"Loaded classes: {[idx_to_class[i] for i in sorted(idx_to_class)]}")

    model = build_model(cfg.model.name, **cfg.model.params).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg.train)

    output_dir = Path(cfg.save.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / cfg.save.best_model_name

    trainer = BaseTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        use_amp=cfg.train.use_amp,
    )
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=cfg.train.epochs,
        best_model_path=str(best_model_path),
    )

    trainer.plot_curves(
        history=history,
        loss_path=str(output_dir / cfg.save.loss_curve_name),
        acc_path=str(output_dir / cfg.save.acc_curve_name),
    )

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(output_dir / "used_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    print("Training finished. Curves and model are saved.")


if __name__ == "__main__":
    # Windows multi-processing safe guard (required when num_workers > 0).
    main()
