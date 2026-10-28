"""Task-neutral transform registry and pipeline assembly.

Transforms operate on a shared sample dictionary. Detection samples provide
``boxes`` and ``cls``; classification-like samples additionally provide a
soft ``target`` vector. Image-only transforms work with either representation.
Every transform is label-aware and deterministic given a seeded ``random.Random``
passed via the pipeline. Transforms needing extra images (Mosaic, MixUp,
CopyPaste) receive the dataset through ``set_dataset``.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

TRANSFORMS: dict[str, type] = {}


def register_transform(cls: type) -> type:
    TRANSFORMS[cls.__name__] = cls
    return cls


class Transform:
    """Base class. Subclasses implement __call__(sample, rng) -> sample."""

    requires_dataset = False

    def set_dataset(self, dataset: Any) -> None:
        self.dataset = dataset

    def __call__(self, sample: dict, rng: random.Random) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


def filter_boxes(sample: dict, original_areas: np.ndarray | None = None, min_visibility: float = 0.1,
                 min_size: float = 2.0) -> dict:
    """Clip boxes to the image and drop degenerate / mostly-lost boxes."""
    h, w = sample["image"].shape[:2]
    boxes = sample["boxes"]
    if boxes.shape[0] == 0:
        return sample
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    keep = (bw > min_size) & (bh > min_size)
    if original_areas is not None:
        area = bw * bh
        keep &= area >= min_visibility * np.maximum(original_areas, 1e-9)
    sample["boxes"] = boxes[keep]
    sample["cls"] = sample["cls"][keep]
    return sample


class Compose:
    def __init__(self, transforms: list[Transform], seed: int | None = None):
        self.transforms = transforms
        self.seed = int(seed or 0)
        self._rngs: dict[tuple[int, int], random.Random] = {}

    def _rng(self) -> random.Random:
        """Return an independent deterministic stream for each data-loader worker."""
        from torch.utils.data import get_worker_info

        info = get_worker_info()
        worker_id = -1 if info is None else info.id
        worker_seed = self.seed if info is None else info.seed
        key = (worker_id, worker_seed)
        if key not in self._rngs:
            self._rngs[key] = random.Random(worker_seed)
        return self._rngs[key]

    def set_dataset(self, dataset: Any) -> None:
        for t in self.transforms:
            if getattr(t, "requires_dataset", False):
                t.set_dataset(dataset)

    def __call__(self, sample: dict, rng_seed: int | None = None, disabled=()) -> dict:
        rng = self._rng() if rng_seed is None else random.Random(rng_seed)
        for t in self.transforms:
            if type(t).__name__ not in disabled:
                sample = t(sample, rng)
        return sample

    def apply_many(self, samples: list[dict]) -> list[dict]:
        """Apply one random transform decision consistently across a sequence.

        This is intended for video frames, paired images, and other inputs where
        independent spatial transforms would destroy alignment.
        """
        if not samples:
            return []
        rng = self._rng()
        initial_state = rng.getstate()
        next_state = None
        output = []
        for sample in samples:
            rng.setstate(initial_state)
            output.append(self(sample))
            if next_state is None:
                next_state = rng.getstate()
        rng.setstate(next_state)
        return output

    def disable(self, names: set[str]) -> None:
        """Zero the probability of named transforms (e.g. close_mosaic)."""
        for t in self.transforms:
            if type(t).__name__ in names and hasattr(t, "p"):
                t.p = 0.0


def build_pipeline(cfg: dict, train: bool = True, seed: int | None = None, *,
                   allow_multisample: bool = True) -> Compose:
    """Assemble the shared train or evaluation pipeline from a config dict."""
    from trainer.augmentation import geometric as g
    from trainer.augmentation import photometric as p
    from trainer.augmentation.config import augmentation_config

    imgsz = int(cfg.get("imgsz", 0))
    cfg = augmentation_config(cfg)
    min_vis = float(cfg.get("min_box_visibility", 0.1))
    if not train:
        return Compose([g.LetterBox(imgsz)] if imgsz > 0 else [], seed=seed)

    profile = cfg.get("aug_profile", "generic")
    degrees = float(cfg["degrees"])
    perspective = float(cfg["perspective"])
    flipud = float(cfg["flipud"])
    fliplr = float(cfg["fliplr"])
    if profile == "topdown":
        degrees = max(degrees, 180.0)
        perspective = 0.0
        flipud = max(flipud, 0.5)
        fliplr = max(fliplr, 0.5)
    elif profile == "frontal":
        degrees = 0.0

    transforms: list[Transform] = []
    if allow_multisample:
        transforms += [
            g.Mosaic(imgsz, p=float(cfg["mosaic"]), p9=float(cfg.get("mosaic9", 0.0))),
            g.CopyPaste(p=float(cfg["copy_paste"]), iou_range=tuple(cfg["copy_paste_iou"])),
        ]
    transforms.append(
        g.RandomPerspective(
            degrees=degrees, translate=float(cfg["translate"]), scale=float(cfg["scale"]),
            shear=float(cfg["shear"]), perspective=perspective, min_visibility=min_vis,
        )
    )
    if allow_multisample:
        transforms.append(g.MixUp(p=float(cfg["mixup"]), alpha=float(cfg["mixup_alpha"])))
    if cfg.get("square_crop"):
        transforms.append(g.SquareCrop(jitter=float(cfg["square_crop_jitter"]), min_visibility=min_vis))
    if imgsz > 0:
        transforms.append(g.LetterBox(imgsz))
    transforms += [
        p.RandomHSV(h=float(cfg["hsv_h"]), s=float(cfg["hsv_s"]), v=float(cfg["hsv_v"])),
        p.BrightnessContrast(brightness=float(cfg["brightness"]), contrast=float(cfg["contrast"]),
                             p=float(cfg["brightness_contrast"])),
        p.MotionBlur(p=float(cfg["blur"])),
        p.GaussNoise(p=float(cfg["noise"])),
        p.JPEGArtifacts(p=float(cfg["jpeg"])),
        p.RandomErasing(p=float(cfg["erasing"])),
        g.RandomFlip(direction="vertical", p=flipud),
        g.RandomFlip(direction="horizontal", p=fliplr),
    ]
    return Compose(transforms, seed=seed)


def pipeline_from_spec(spec: list[dict], seed: int | None = None) -> Compose:
    """Build a fully custom pipeline from a TOML array of {name, params} tables."""
    transforms = []
    for item in spec:
        item = dict(item)
        name = item.pop("name")
        if name not in TRANSFORMS:
            raise KeyError(f"unknown transform '{name}'; registered: {sorted(TRANSFORMS)}")
        transforms.append(TRANSFORMS[name](**item))
    return Compose(transforms, seed=seed)
