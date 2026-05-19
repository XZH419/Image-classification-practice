from __future__ import annotations

"""Training loop utilities for supervised image classification.

``BaseTrainer`` owns the epoch loop, automatic mixed precision on CUDA,
metric accumulation, best-checkpoint saving by validation accuracy, and
learning-curve plotting.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader


class BaseTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        use_amp: bool = True,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

    def _run_one_epoch(self, loader: DataLoader, training: bool) -> Tuple[float, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                with autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            preds = outputs.argmax(dim=1)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            correct += (preds == targets).sum().item()
            total += batch_size

        epoch_loss = total_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int,
        best_model_path: str,
    ) -> Dict[str, List[float]]:
        history = {
            "train_loss": [],
            "train_acc": [],
            "valid_loss": [],
            "valid_acc": [],
        }
        best_valid_acc = 0.0
        save_path = Path(best_model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_one_epoch(train_loader, training=True)
            valid_loss, valid_acc = self._run_one_epoch(valid_loader, training=False)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["valid_loss"].append(valid_loss)
            history["valid_acc"].append(valid_acc)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(self.model.state_dict(), str(save_path))

            print(
                f"[Epoch {epoch:02d}/{epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}"
            )

        print(f"Best validation accuracy: {best_valid_acc:.4f}")
        print(f"Best model saved to: {save_path}")
        return history

    @staticmethod
    def plot_curves(history: Dict[str, List[float]], loss_path: str, acc_path: str) -> None:
        Path(loss_path).parent.mkdir(parents=True, exist_ok=True)
        epochs = range(1, len(history["train_loss"]) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["train_loss"], label="Train Loss")
        plt.plot(epochs, history["valid_loss"], label="Valid Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(loss_path, dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["train_acc"], label="Train Accuracy")
        plt.plot(epochs, history["valid_acc"], label="Valid Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(acc_path, dpi=200)
        plt.close()
