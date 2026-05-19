from __future__ import annotations

"""Evaluation utilities for final STL-10 test-set reporting.

The evaluator runs inference, stores true/predicted labels, builds the sklearn
classification report, and renders the confusion matrix as a heatmap.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader


class Evaluator:
    def __init__(self, model: torch.nn.Module, device: torch.device, idx_to_class: Dict[int, str]) -> None:
        self.model = model
        self.device = device
        self.idx_to_class = idx_to_class

    @torch.no_grad()
    def infer(self, data_loader: DataLoader) -> Dict[str, List[int]]:
        self.model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []

        for inputs, labels in data_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model(inputs)
            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

        return {"y_true": y_true, "y_pred": y_pred}

    def build_report(self, y_true: List[int], y_pred: List[int]) -> str:
        class_names = [self.idx_to_class[i] for i in sorted(self.idx_to_class)]
        return classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )

    def save_confusion_matrix(self, y_true: List[int], y_pred: List[int], output_path: str) -> None:
        class_names = [self.idx_to_class[i] for i in sorted(self.idx_to_class)]
        cm = confusion_matrix(y_true, y_pred, labels=sorted(self.idx_to_class))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
