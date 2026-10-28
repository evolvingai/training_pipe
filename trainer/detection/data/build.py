"""Dataset TOML parsing and dataloader construction."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from trainer.augmentation import build_pipeline
from trainer.detection.config import load_toml
from trainer.detection.data.coco_dataset import CocoDetectionDataset
from trainer.detection.data.dataset import IMG_SUFFIXES, DetectionDataset
from trainer.detection.data.epoch_sampler import EpochSampler


def _is_coco_spec(spec) -> bool:
    return isinstance(spec, dict) and spec.get("format") == "coco"


def parse_dataset_toml(path: str | Path) -> dict:
    """Parse and validate a dataset descriptor.

    Required: ``train``, ``val``. ``names`` is required for YOLO-format splits
    and optional for COCO-format splits (derived from the JSON categories).
    Optional: ``path`` root, ``test``.

    YOLO split specs: a directory, a .txt file of image paths, a list, or a
    list of weighted sources ``[{path = "...", weight = 1.0}, ...]``.

    COCO split spec (inline table)::

        [train]
        format      = "coco"
        images      = "images/train"
        annotations = "annotations/instances_train.json"
    """
    path = Path(path)
    data = load_toml(path)
    for key in ("train", "val"):
        if key not in data:
            raise KeyError(f"dataset toml missing required key '{key}': {path}")

    root = Path(data.get("path", path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()

    if "names" in data:
        names = {int(k): str(v) for k, v in data["names"].items()}
    else:
        # Derive class names from the first COCO split's categories array.
        names = None
        for split in ("train", "val", "test"):
            if _is_coco_spec(data.get(split)):
                ann = Path(data[split]["annotations"])
                if not ann.is_absolute():
                    ann = root / ann
                coco_data = json.loads(ann.read_text())
                cats = sorted(coco_data.get("categories", []), key=lambda c: c["id"])
                names = {i: c["name"] for i, c in enumerate(cats)}
                break
        if names is None:
            raise KeyError(f"dataset toml missing required key 'names': {path}")

    if not names:
        raise ValueError(f"dataset must define at least one class: {path}")
    expected_ids = list(range(len(names)))
    if sorted(names) != expected_ids:
        raise ValueError(f"class ids must be contiguous from 0; got {sorted(names)} in {path}")
    if any(not name.strip() for name in names.values()):
        raise ValueError(f"class names must be non-empty: {path}")
    if len(set(names.values())) != len(names):
        raise ValueError(f"class names must be unique: {path}")

    coco_category_ids: list[int] | None = None
    for split in ("train", "val", "test"):
        spec = data.get(split)
        if not _is_coco_spec(spec):
            continue
        missing = {key for key in ("images", "annotations") if key not in spec}
        if missing:
            raise KeyError(f"COCO split '{split}' missing keys: {sorted(missing)}")
        ann = Path(spec["annotations"])
        if not ann.is_absolute():
            ann = root / ann
        coco_data = json.loads(ann.read_text())
        category_ids = sorted(int(c["id"]) for c in coco_data.get("categories", []))
        if len(category_ids) != len(set(category_ids)):
            raise ValueError(f"duplicate COCO category ids in {ann}")
        if coco_category_ids is None:
            coco_category_ids = category_ids
        elif category_ids != coco_category_ids:
            raise ValueError(f"COCO category ids differ between dataset splits: {ann}")
    if coco_category_ids is not None and len(coco_category_ids) != len(names):
        raise ValueError(
            f"dataset defines {len(names)} names but COCO annotations define "
            f"{len(coco_category_ids)} categories"
        )

    out = {"root": root, "names": names, "nc": len(names)}
    for split in ("train", "val", "test"):
        if split in data:
            out[split] = data[split]
    return out


def _resolve_split(spec, root: Path) -> tuple[list[str], list[float] | None]:
    """Resolve a split spec to (image paths, optional per-image source weights)."""
    if isinstance(spec, list) and spec and isinstance(spec[0], dict):
        paths: list[str] = []
        weights: list[float] = []
        for source in spec:
            sub_paths, _ = _resolve_split(source["path"], root)
            w = float(source.get("weight", 1.0))
            paths.extend(sub_paths)
            weights.extend([w] * len(sub_paths))
        return paths, weights
    if isinstance(spec, list):
        paths = []
        for s in spec:
            sub, _ = _resolve_split(s, root)
            paths.extend(sub)
        return paths, None
    p = Path(spec)
    if not p.is_absolute():
        p = root / p
    if p.is_dir():
        # os.walk avoids Path object creation per entry, significantly faster than rglob on large datasets
        collected = []
        for dirpath, _, filenames in os.walk(p):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in IMG_SUFFIXES:
                    collected.append(os.path.join(dirpath, fn))
        return sorted(collected), None
    if p.suffix == ".txt":
        lines = [line.strip() for line in p.read_text().splitlines() if line.strip()]
        return [line if Path(line).is_absolute() else str(root / line) for line in lines], None
    raise FileNotFoundError(f"cannot resolve dataset split: {spec}")


def collate_fn(batch: list[dict]) -> dict:
    imgs = torch.stack([b["img"] for b in batch])
    cls_list, box_list, idx_list = [], [], []
    for i, b in enumerate(batch):
        n = b["cls"].shape[0]
        cls_list.append(b["cls"])
        box_list.append(b["bboxes"])
        idx_list.append(torch.full((n,), i, dtype=torch.long))
    return {
        "img": imgs,
        "cls": torch.cat(cls_list) if cls_list else torch.zeros(0),
        "bboxes": torch.cat(box_list) if box_list else torch.zeros(0, 4),
        "batch_idx": torch.cat(idx_list) if idx_list else torch.zeros(0, dtype=torch.long),
        "paths": [b["path"] for b in batch],
        "orig_shapes": [b["orig_shape"] for b in batch],
        "ratios": [b["ratio"] for b in batch],
        "pads": [b["pad"] for b in batch],
    }


def build_class_weights(dataset: DetectionDataset, nc: int) -> np.ndarray:
    """Per-image sampling weights: mean inverse frequency of contained classes."""
    counts = dataset.class_counts(nc).astype(np.float64)
    inv_freq = 1.0 / np.maximum(counts / max(counts.sum(), 1), 1e-12)

    label_counts = np.array([len(lbl) for lbl in dataset.labels], dtype=np.int64)
    if label_counts.sum() == 0:
        return np.ones(len(dataset))

    nonempty = np.where(label_counts > 0)[0]
    img_idx = np.repeat(nonempty, label_counts[nonempty])
    all_cls = np.concatenate(
        [dataset.labels[i][:, 0].astype(int).clip(0, nc - 1) for i in nonempty]
    )
    weight_sum = np.zeros(len(dataset))
    np.add.at(weight_sum, img_idx, inv_freq[all_cls])
    weights = np.ones(len(dataset))
    weights[nonempty] = weight_sum[nonempty] / label_counts[nonempty]
    return weights


def _build_coco_dataset(
    spec: dict, root: Path, cfg: dict, train: bool, seed_offset: int = 0,
) -> CocoDetectionDataset:
    images_dir = Path(spec["images"])
    if not images_dir.is_absolute():
        images_dir = root / images_dir
    ann = Path(spec["annotations"])
    if not ann.is_absolute():
        ann = root / ann
    transforms = build_pipeline(cfg, train=train, seed=int(cfg["seed"]) + seed_offset)
    return CocoDetectionDataset(
        images_dir,
        ann,
        imgsz=int(cfg["imgsz"]),
        transforms=transforms,
        cache=str(cfg.get("cache", "none")),
        fraction=float(cfg.get("fraction", 1.0)) if train else 1.0,
    )


class DistributedWeightedSampler(Sampler[int]):
    """Shard one deterministic weighted draw stream across DDP ranks."""

    def __init__(self, weights, num_replicas: int, rank: int, seed: int = 0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(np.ceil(len(self.weights) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, replacement=True, generator=generator)
        return iter(indices[self.rank:self.total_size:self.num_replicas].tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def build_dataloader(data: dict, cfg: dict, split: str = "train", batch: int | None = None,
                     seed_offset: int = 0, rank: int = 0, world_size: int = 1) -> DataLoader:
    train = split == "train"
    spec = data[split]
    source_weights = None

    if _is_coco_spec(spec):
        dataset = _build_coco_dataset(spec, data["root"], cfg, train, seed_offset=seed_offset)
    else:
        paths, source_weights = _resolve_split(spec, data["root"])
        transforms = build_pipeline(cfg, train=train, seed=int(cfg["seed"]) + seed_offset)
        dataset = DetectionDataset(
            paths, imgsz=int(cfg["imgsz"]), transforms=transforms,
            cache=str(cfg.get("cache", "none")), fraction=float(cfg.get("fraction", 1.0)) if train else 1.0,
            nc=int(data["nc"]),
        )

    batch = batch or int(cfg["batch"])
    workers = int(cfg["workers"])
    sampler = None
    shuffle = train
    weights = None
    if train and cfg.get("balanced_sampling"):
        weights = build_class_weights(dataset, data["nc"])
    if source_weights is not None:
        by_path = dict(zip(paths, source_weights))
        sw = np.array([by_path.get(p, 1.0) for p in dataset.paths])
        weights = sw if weights is None else weights * sw
    generator = torch.Generator().manual_seed(
        int(cfg["seed"]) + seed_offset + (0 if train else 1_000_000)
    )
    if train and weights is not None:
        sampler = DistributedWeightedSampler(weights, world_size, rank, seed=int(cfg["seed"]))
        shuffle = False
    elif train:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=train, seed=int(cfg["seed"]),
            drop_last=train and len(dataset) > batch,
        )
        shuffle = False
    if train:
        sampler = EpochSampler(sampler, int(cfg["seed"]), rank, int(cfg["epochs"]), int(cfg["close_mosaic"]))
    effective_length = len(sampler) if sampler is not None else len(dataset)
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=int(cfg.get("prefetch_factor", 4)) if workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=train and effective_length > batch,
        generator=generator,
    )
