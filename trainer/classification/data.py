"""Folder-based image classification datasets used by Trainer."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from trainer.augmentation import (
    CLASSIFICATION_AUGMENTATION_DEFAULTS,
    augmentation_config,
    build_pipeline,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def select_device(value: str) -> torch.device:
    value = value.strip().lower()
    if value in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def image_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


def class_names(folder: Path) -> list[str]:
    if not folder.is_dir():
        raise FileNotFoundError(f"dataset folder does not exist: {folder}")
    names = sorted(entry.name for entry in folder.iterdir() if entry.is_dir() and image_files(entry))
    if len(names) < 2:
        raise ValueError(f"classifier needs at least two non-empty class folders under {folder}")
    return names


def samples_for(folder: Path, names: list[str]) -> list[Sample]:
    output: list[Sample] = []
    for label, name in enumerate(names):
        target = folder / name
        if target.is_dir():
            output.extend(Sample(path, label) for path in image_files(target))
    return output


def discover_training(root: Path, val_fraction: float, seed: int) -> tuple[list[Sample], list[Sample], list[str]]:
    train_root = root / "train"
    val_root = root / "val"
    if train_root.is_dir():
        names = class_names(train_root)
        train = samples_for(train_root, names)
        if not val_root.is_dir():
            raise FileNotFoundError(f"{train_root} exists but validation folder is missing: {val_root}")
        val = samples_for(val_root, names)
    else:
        names = class_names(root)
        train, val = [], []
        rng = random.Random(seed)
        for label, name in enumerate(names):
            paths = image_files(root / name)
            rng.shuffle(paths)
            val_count = max(1, round(len(paths) * val_fraction)) if len(paths) > 1 else 0
            val.extend(Sample(path, label) for path in paths[:val_count])
            train.extend(Sample(path, label) for path in paths[val_count:])
    if not train or not val:
        raise ValueError("training and validation splits must both contain images")
    return train, val, names


def discover_evaluation(root: Path, names: list[str]) -> list[Sample]:
    folder = root / "val" if (root / "val").is_dir() else root
    samples = samples_for(folder, names)
    if not samples:
        raise ValueError(f"no labelled images found under {folder}")
    return samples


class ImageDataset(Dataset):
    """Classification dataset backed by the task-neutral augmentation pipeline."""

    def __init__(self, samples: list[Sample], imgsz: int, augment: bool = False, *,
                 num_classes: int | None = None, augmentation: dict | None = None,
                 augmentation_config_path: str | Path | None = None, seed: int = 0,
                 transforms=None):
        self.samples = samples
        self.imgsz = imgsz
        self.augment = augment
        self.num_classes = num_classes or (max((sample.label for sample in samples), default=-1) + 1)
        if transforms is None:
            config = augmentation_config(
                augmentation, defaults=CLASSIFICATION_AUGMENTATION_DEFAULTS,
                config_path=augmentation_config_path,
            )
            config["imgsz"] = imgsz
            transforms = build_pipeline(config, train=augment, seed=seed)
        self.transforms = transforms
        if hasattr(self.transforms, "set_dataset"):
            self.transforms.set_dataset(self)

    def __len__(self) -> int:
        return len(self.samples)

    def load_sample(self, index: int) -> dict:
        sample = self.samples[index]
        image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read image: {sample.path}")
        target = np.zeros(self.num_classes, dtype=np.float32)
        target[sample.label] = 1.0
        return {
            "image": image,
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "cls": np.zeros((0,), dtype=np.float32),
            "target": target,
            "label": sample.label,
            "path": str(sample.path),
        }

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int | torch.Tensor, str]:
        raw = self.load_sample(index)
        transformed = self.transforms(raw) if self.transforms is not None else raw
        image = cv2.cvtColor(transformed["image"], cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)
        tensor = tensor.sub_(0.5).div_(0.25)
        target: int | torch.Tensor = (
            torch.from_numpy(np.asarray(transformed["target"], dtype=np.float32))
            if self.augment else int(raw["label"])
        )
        return tensor, target, str(raw["path"])


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels), nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels), nn.SiLU(inplace=True),
        )


class TinyClassifier(nn.Module):
    """Legacy compact classifier retained for v1 checkpoint loading and tests."""

    def __init__(self, classes: int, width: int = 32, dropout: float = 0.15):
        super().__init__()
        channels = [width, width * 2, width * 4, width * 6]
        blocks: list[nn.Module] = []
        previous = 3
        for current in channels:
            blocks.append(ConvBlock(previous, current))
            previous = current
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels[-1], classes))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(image)))


def make_loader(samples: list[Sample], imgsz: int, batch: int, augment: bool, workers: int,
                shuffle: bool | None = None, *, num_classes: int | None = None,
                augmentation: dict | None = None, augmentation_config_path: str | Path | None = None,
                seed: int = 0) -> DataLoader:
    return DataLoader(
        ImageDataset(
            samples, imgsz, augment=augment, num_classes=num_classes, augmentation=augmentation,
            augmentation_config_path=augmentation_config_path, seed=seed,
        ), batch_size=batch,
        shuffle=augment if shuffle is None else shuffle, num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


__all__ = [
    "IMAGE_SUFFIXES", "ConvBlock", "ImageDataset", "Sample", "TinyClassifier", "class_names",
    "discover_evaluation", "discover_training", "image_files", "make_loader", "samples_for", "select_device",
]
