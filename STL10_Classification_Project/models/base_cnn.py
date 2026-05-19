from __future__ import annotations

"""Baseline convolutional classifier for 96x96 STL-10 images.

The model follows the assignment requirement directly: three convolutional
blocks, each followed by ReLU and max pooling, then a two-layer classifier.
Spatial resolution shrinks from 96 to 12 while channel capacity grows from
32 to 128, matching a compact VGG-style design.
"""

import torch
import torch.nn as nn

from models.registry import register_model


@register_model("BaseCNN")
class BaseCNN(nn.Module):
    """Three-block CNN used as the baseline experiment.

    Input:  3 x 96 x 96
    Block1: 32 x 48 x 48
    Block2: 64 x 24 x 24
    Block3: 128 x 12 x 12
    FC: 18432 -> 512 -> num_classes
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
