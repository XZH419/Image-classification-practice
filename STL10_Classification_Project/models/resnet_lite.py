from __future__ import annotations

"""A lightweight handwritten residual network for architecture comparison.

This optional experiment keeps the model small enough for STL-10 coursework
while introducing residual shortcuts, batch normalization, downsampling blocks,
and global average pooling.
"""

import torch
import torch.nn as nn

from models.registry import register_model


class ResidualBlock(nn.Module):
    """Two-convolution residual block with an optional projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return self.relu(out)


@register_model("ResNetLite")
class ResNetLite(nn.Module):
    def __init__(self, num_classes: int = 10, base_channels: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(base_channels, base_channels, stride=1)
        self.layer2 = ResidualBlock(base_channels, base_channels * 2, stride=2)
        self.layer3 = ResidualBlock(base_channels * 2, base_channels * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
