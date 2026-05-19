from __future__ import annotations

"""Dataset and DataLoader builders for the STL-10 folder layout.

Training code must not touch the held-out test split. This module therefore
has separate builders: one that splits ``data/train`` into train/validation
subsets, and another that loads ``data/test`` only for final evaluation.
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _build_transforms(image_size: int, use_aug: bool = False) -> Tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    base_ops = [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]

    if use_aug:
        train_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_tf = transforms.Compose(base_ops)

    valid_tf = transforms.Compose(base_ops)
    return train_tf, valid_tf


def build_train_valid_loaders(
    train_dir: str,
    batch_size: int,
    image_size: int = 96,
    train_size: int = 5600,
    valid_size: int = 1400,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    use_aug: bool = False,
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    train_root = Path(train_dir)
    if not train_root.exists():
        raise FileNotFoundError(f"Training folder not found: {train_root.resolve()}")

    train_tf, valid_tf = _build_transforms(image_size=image_size, use_aug=use_aug)
    train_dataset = datasets.ImageFolder(root=str(train_root), transform=train_tf)
    valid_dataset = datasets.ImageFolder(root=str(train_root), transform=valid_tf)

    expected_total = train_size + valid_size
    if len(train_dataset) != expected_total:
        raise ValueError(
            f"Expected {expected_total} images in training set, got {len(train_dataset)} from {train_root}."
        )

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(expected_total, generator=generator).tolist()
    train_subset = Subset(train_dataset, shuffled_indices[:train_size])
    valid_subset = Subset(valid_dataset, shuffled_indices[train_size:])

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    return train_loader, valid_loader, idx_to_class


def build_test_loader(
    test_dir: str,
    batch_size: int,
    image_size: int = 96,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Dict[int, str]]:
    test_root = Path(test_dir)
    if not test_root.exists():
        raise FileNotFoundError(f"Test folder not found: {test_root.resolve()}")

    _, test_tf = _build_transforms(image_size=image_size, use_aug=False)
    test_dataset = datasets.ImageFolder(root=str(test_root), transform=test_tf)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    return test_loader, idx_to_class
