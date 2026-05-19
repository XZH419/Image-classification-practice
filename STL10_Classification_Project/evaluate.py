from __future__ import annotations

"""Evaluate a saved STL-10 classifier on the held-out test folder.

This file intentionally uses only ``data/test`` and a previously saved
checkpoint. It reports per-class precision, recall, F1-score, macro averages,
and a 10-class confusion matrix for the final model assessment.
"""

import argparse
from pathlib import Path

import torch

from engine.evaluator import Evaluator
from models import base_cnn, resnet_lite  # noqa: F401  # trigger model registration
from models.registry import build_model
from train import load_config, resolve_device
from utils.data_loader import build_test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate STL-10 classifier on test split")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to checkpoint (.pth). Defaults to config save path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.device)

    output_dir = Path(cfg.save.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = Path(args.weights) if args.weights else output_dir / cfg.save.best_model_name
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path.resolve()}")

    model = build_model(cfg.model.name, **cfg.model.params).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    test_loader, idx_to_class = build_test_loader(
        test_dir=cfg.data.test_dir,
        batch_size=cfg.train.batch_size,
        image_size=cfg.data.image_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    evaluator = Evaluator(model=model, device=device, idx_to_class=idx_to_class)
    outputs = evaluator.infer(test_loader)

    report = evaluator.build_report(outputs["y_true"], outputs["y_pred"])
    print(report)

    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    cm_path = output_dir / "confusion_matrix.png"
    evaluator.save_confusion_matrix(outputs["y_true"], outputs["y_pred"], str(cm_path))

    print(f"Classification report saved to: {report_path}")
    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
