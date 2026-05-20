from __future__ import annotations

"""Generate Grad-CAM visual explanations for a trained STL-10 classifier.

The script loads a saved checkpoint, selects test images from each class, and
overlays Grad-CAM heatmaps on the original images. The outputs help inspect
which regions drive the model decision for correct and incorrect predictions.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from models import base_cnn, resnet_lite  # noqa: F401  # trigger model registration
from models.base_cnn import BaseCNN
from models.registry import build_model
from models.resnet_lite import ResNetLite
from train import load_config, resolve_device
from utils.cam_viz import GradCAM, save_cam_overlay
from utils.data_loader import _build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--weights", type=str, default=None, help="Optional checkpoint path.")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=1,
        help="Number of test images visualized for each class.",
    )
    return parser.parse_args()


def pick_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Choose the last convolutional layer for common project models."""
    if isinstance(model, BaseCNN):
        return model.features[6]
    if isinstance(model, ResNetLite):
        return model.layer3.conv2
    conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    if not conv_layers:
        raise ValueError("Grad-CAM requires at least one Conv2d layer.")
    return conv_layers[-1]


def disable_inplace_relu(model: torch.nn.Module) -> None:
    """Avoid Grad-CAM backward-hook conflicts with in-place activations."""
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized CHW tensor in [-1, 1] back to an HWC RGB array."""
    image = image_tensor.detach().cpu().clone()
    image = image * 0.5 + 0.5
    image = image.clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def collect_balanced_indices(dataset: datasets.ImageFolder, samples_per_class: int) -> List[int]:
    class_counts: Dict[int, int] = {idx: 0 for idx in range(len(dataset.classes))}
    selected: List[int] = []
    for idx, (_, label) in enumerate(dataset.samples):
        if class_counts[label] < samples_per_class:
            selected.append(idx)
            class_counts[label] += 1
        if all(count >= samples_per_class for count in class_counts.values()):
            break
    return selected


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.device)

    output_dir = Path(cfg.save.output_dir) / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = Path(args.weights) if args.weights else Path(cfg.save.output_dir) / cfg.save.best_model_name
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path.resolve()}")

    model = build_model(cfg.model.name, **cfg.model.params).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    disable_inplace_relu(model)
    model.eval()

    _, test_tf = _build_transforms(image_size=cfg.data.image_size, use_aug=False)
    test_dataset = datasets.ImageFolder(root=cfg.data.test_dir, transform=test_tf)
    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    selected_indices = collect_balanced_indices(test_dataset, args.samples_per_class)
    selected_loader = DataLoader(Subset(test_dataset, selected_indices), batch_size=1, shuffle=False)

    gradcam = GradCAM(model=model, target_layer=pick_target_layer(model))
    rows: List[Tuple[str, str, str, str, str]] = []

    for local_idx, (image_tensor, label_tensor) in enumerate(selected_loader):
        image_tensor = image_tensor.to(device)
        label = int(label_tensor.item())

        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, pred].item())

        cam = gradcam.generate(image_tensor, class_idx=pred)
        image_np = denormalize(image_tensor[0])

        true_name = idx_to_class[label]
        pred_name = idx_to_class[pred]
        status = "correct" if pred == label else "wrong"
        filename = f"{local_idx:02d}_{true_name}_pred-{pred_name}_{status}.png"
        save_cam_overlay(image_np, cam, str(output_dir / filename), alpha=0.45)

        rows.append((filename, true_name, pred_name, f"{confidence:.4f}", status))

    summary_path = output_dir / "gradcam_summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "true_label", "pred_label", "confidence", "status"])
        writer.writerows(rows)

    print(f"Generated {len(rows)} Grad-CAM overlays in: {output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
