"""Detection dataset: binary label cache, optional RAM/disk image cache.

Labels: one txt per image, lines of ``cls cx cy w h`` normalized to [0, 1].
Label txt path is derived from the image path by replacing the last
``/images/`` path component with ``/labels/`` and the suffix with ``.txt``.
"""

from __future__ import annotations

import hashlib
import concurrent.futures as cf
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from trainer.detection.utils.image import imread_rgb
from trainer.detection.utils.logging import LOGGER

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CACHE_VERSION = "3"  # bumped for the BGR -> RGB switch; invalidates cached pixels


def img2label_path(img_path: str) -> str:
    p = Path(img_path)
    parts = list(p.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return str(Path(*parts).with_suffix(".txt"))


def _parse_one(
    img_path: str, nc: int | None = None,
) -> tuple[str, tuple[int, int] | None, np.ndarray | None, str | None]:
    """Validate one image and parse its label file. Returns (path, (h, w), labels, error)."""
    img = cv2.imread(img_path)  # channel order is irrelevant here: only the shape is used
    if img is None:
        return img_path, None, None, "unreadable image"
    h, w = img.shape[:2]
    lbl_path = img2label_path(img_path)
    labels = np.zeros((0, 5), dtype=np.float32)
    if Path(lbl_path).exists():
        try:
            rows = []
            for line in Path(lbl_path).read_text().strip().splitlines():
                vals = line.split()
                if len(vals) != 5:
                    return img_path, None, None, f"bad label row in {lbl_path}: {line!r}"
                rows.append([float(v) for v in vals])
            if rows:
                labels = np.array(rows, dtype=np.float32)
                if not np.isfinite(labels).all():
                    return img_path, None, None, f"non-finite label value in {lbl_path}"
                cls = labels[:, 0]
                if not np.equal(cls, np.floor(cls)).all():
                    return img_path, None, None, f"class ids must be integers in {lbl_path}"
                if (cls < 0).any() or (nc is not None and (cls >= nc).any()):
                    expected = f"[0, {nc - 1}]" if nc is not None else "non-negative"
                    return img_path, None, None, f"class id outside {expected} in {lbl_path}"
                coords = labels[:, 1:]
                if coords.max() > 1.001 or coords.min() < -0.001:
                    return img_path, None, None, f"non-normalized coords in {lbl_path}"
                if (coords[:, 2:] <= 0).any():
                    return img_path, None, None, f"box width and height must be positive in {lbl_path}"
                x1y1 = coords[:, :2] - coords[:, 2:] / 2
                x2y2 = coords[:, :2] + coords[:, 2:] / 2
                if (x1y1 < -0.001).any() or (x2y2 > 1.001).any():
                    return img_path, None, None, f"box extends outside image in {lbl_path}"
        except ValueError as e:
            return img_path, None, None, f"unparseable label {lbl_path}: {e}"
    return img_path, (h, w), labels, None


class DetectionDataset(Dataset):
    """Loads images + labels with a binary cache keyed by file list + mtimes."""

    def __init__(self, img_paths: list[str], imgsz: int = 640, transforms=None,
                 cache: str = "none", fraction: float = 1.0, nc: int | None = None):
        img_paths = sorted(p for p in img_paths if Path(p).suffix.lower() in IMG_SUFFIXES)
        if fraction < 1.0:
            img_paths = img_paths[: max(1, int(len(img_paths) * fraction))]
        if not img_paths:
            raise FileNotFoundError("no images found for dataset")
        self.imgsz = imgsz
        self.nc = nc
        self.transforms = transforms
        self.cache_mode = cache
        self.paths, self.shapes, self.labels = self._load_or_build_cache(img_paths)
        self._ram: list[np.ndarray | None] = [None] * len(self.paths)
        self.skipped_total = len(img_paths) - len(self.paths)
        if self.cache_mode == "disk":
            self._build_disk_cache()
        if self.cache_mode == "ram":
            self._preload_ram_cache()
        if transforms is not None and hasattr(transforms, "set_dataset"):
            transforms.set_dataset(self)

    # -- cache ---------------------------------------------------------------
    def _cache_key(self, img_paths: list[str]) -> str:
        def _stat_entry(p: str) -> str:
            st = Path(p).stat()
            entry = f"{p}:{st.st_mtime_ns}:{st.st_size}"
            lbl = Path(img2label_path(p))
            if lbl.exists():
                lst = lbl.stat()
                entry += f"|{lbl}:{lst.st_mtime_ns}:{lst.st_size}"
            return entry

        hasher = hashlib.sha256()
        hasher.update(CACHE_VERSION.encode())
        with cf.ThreadPoolExecutor() as pool:
            for entry in pool.map(_stat_entry, img_paths):
                hasher.update(entry.encode())
        return hasher.hexdigest()[:16]

    def _load_or_build_cache(self, img_paths: list[str]):
        cache_path = Path(img_paths[0]).parent / f".trainer_cache_{self._cache_key(img_paths)}.npz"
        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    paths = [str(p) for p in data["paths"]]
                    shapes = data["shapes"]
                    flat, splits = data["labels_flat"], data["labels_splits"]
                    labels = [flat[a:b].reshape(-1, 5) for a, b in zip(splits[:-1], splits[1:])]
                if self.nc is None or all(
                    not len(lb) or ((lb[:, 0] >= 0) & (lb[:, 0] < self.nc)).all() for lb in labels
                ):
                    LOGGER.info(f"dataset cache hit: {len(paths)} images ({cache_path.name})")
                    return paths, shapes, labels
                LOGGER.warning(f"dataset cache has class ids outside [0, {self.nc - 1}]; rebuilding")
            except (OSError, ValueError, KeyError) as error:
                LOGGER.warning(f"ignoring invalid dataset cache {cache_path}: {error}")
        total = len(img_paths)
        LOGGER.info(f"scanning dataset ({total} images)...")
        paths, shapes, labels = [], [], []
        skipped = 0
        deferred_warnings: list[str] = []
        with cf.ThreadPoolExecutor() as pool:
            parser = partial(_parse_one, nc=self.nc)
            for done, (img_path, shape, lbl, err) in enumerate(pool.map(parser, img_paths), 1):
                if err is not None:
                    deferred_warnings.append(f"skipping {img_path}: {err}")
                    skipped += 1
                else:
                    paths.append(img_path)
                    shapes.append(shape)
                    labels.append(lbl)
                filled = int(25 * done / total)
                bar = '█' * filled + '░' * (25 - filled)
                print(f"\r  [{bar}] {done}/{total} images, {total - done} remaining", end='', flush=True)
        print(flush=True)
        for w in deferred_warnings:
            LOGGER.warning(w)
        if not paths:
            raise RuntimeError("all dataset samples were invalid")
        LOGGER.info(f"scan complete: {len(paths)}/{total} valid{f', {skipped} skipped' if skipped else ''}")
        if skipped:
            LOGGER.warning(f"dataset scan skipped {skipped} invalid samples")
        shapes_arr = np.array(shapes, dtype=np.int32)
        flat = np.concatenate([lb.reshape(-1) for lb in labels]) if labels else np.zeros(0, np.float32)
        splits = np.cumsum([0] + [lb.size for lb in labels])
        try:
            np.savez(cache_path, paths=np.array(paths), shapes=shapes_arr,
                     labels_flat=flat.astype(np.float32), labels_splits=splits.astype(np.int64))
        except OSError as e:  # read-only dataset dir: run without cache
            LOGGER.warning(f"could not write dataset cache: {e}")
        return paths, shapes_arr, labels

    def _disk_cache_path(self, idx: int) -> Path:
        image_path = Path(self.paths[idx])
        stat = image_path.stat()
        token = hashlib.sha256(
            f"{CACHE_VERSION}:{self.imgsz}:{stat.st_mtime_ns}:{stat.st_size}".encode()
        ).hexdigest()[:10]
        return image_path.with_suffix(f".trainer_{self.imgsz}_{token}.npy")

    def _build_disk_cache(self) -> None:
        total = len(self.paths)

        def _cache_one(i: int) -> None:
            p = self._disk_cache_path(i)
            if not p.exists():
                try:
                    np.save(p, self._decode(i))
                except OSError as error:
                    LOGGER.warning(f"could not write image cache {p}: {error}")

        with cf.ThreadPoolExecutor() as pool:
            for done, _ in enumerate(pool.map(_cache_one, range(total)), 1):
                filled = int(25 * done / total)
                bar = '█' * filled + '░' * (25 - filled)
                print(
                    f"\r  caching  [{bar}] {done}/{total} images, {total - done} remaining",
                    end="", flush=True,
                )
        print()

    def _preload_ram_cache(self) -> None:
        total = len(self.paths)
        LOGGER.info(f"preloading {total} images into RAM cache...")

        def _safe_decode(i: int) -> np.ndarray | None:
            try:
                return self._decode(i)
            except Exception as e:
                LOGGER.warning(f"could not cache {self.paths[i]}: {e}")
                return None

        with cf.ThreadPoolExecutor() as pool:
            for done, (i, img) in enumerate(
                zip(range(total), pool.map(_safe_decode, range(total))), 1
            ):
                self._ram[i] = img
                filled = int(25 * done / total)
                bar = '█' * filled + '░' * (25 - filled)
                print(f"\r  RAM cache  [{bar}] {done}/{total}", end="", flush=True)
        print()

    # -- loading -------------------------------------------------------------
    def _decode(self, idx: int) -> np.ndarray:
        """Decode to RGB and pre-resize so the long side == imgsz (cache-friendly)."""
        img = imread_rgb(self.paths[idx])
        if img is None:
            raise RuntimeError(f"image became unreadable: {self.paths[idx]}")
        h, w = img.shape[:2]
        r = self.imgsz / max(h, w)
        if r != 1.0:
            img = cv2.resize(img, (round(w * r), round(h * r)),
                             interpolation=cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA)
        return img

    def _get_image(self, idx: int) -> np.ndarray:
        if self.cache_mode == "ram":
            if self._ram[idx] is None:
                self._ram[idx] = self._decode(idx)
            return self._ram[idx].copy()
        if self.cache_mode == "disk":
            p = self._disk_cache_path(idx)
            if p.exists():
                try:
                    return np.load(p, allow_pickle=False)
                except (OSError, ValueError) as error:
                    LOGGER.warning(f"ignoring invalid image cache {p}: {error}")
        return self._decode(idx)

    def load_sample(self, idx: int) -> dict:
        """Raw sample: image (RGB, pre-resized), boxes xyxy pixels, cls. No transforms."""
        img = self._get_image(idx)
        h, w = img.shape[:2]
        lbl = self.labels[idx]
        cls = lbl[:, 0].copy()
        boxes = np.zeros((0, 4), dtype=np.float32)
        if lbl.shape[0]:
            cxcywh = lbl[:, 1:].copy()
            boxes = np.stack([
                (cxcywh[:, 0] - cxcywh[:, 2] / 2) * w,
                (cxcywh[:, 1] - cxcywh[:, 3] / 2) * h,
                (cxcywh[:, 0] + cxcywh[:, 2] / 2) * w,
                (cxcywh[:, 1] + cxcywh[:, 3] / 2) * h,
            ], 1).astype(np.float32)
        return {"image": img, "boxes": boxes, "cls": cls.astype(np.float32), "path": self.paths[idx],
                "orig_shape": tuple(self.shapes[idx])}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int | tuple[int, int, bool]) -> dict:
        seed, closed = None, False
        if isinstance(idx, tuple):
            idx, seed, closed = idx
        sample = self.load_sample(idx)
        if self.transforms is not None:
            if seed is None:
                sample = self.transforms(sample)
            else:
                sample = self.transforms(sample, rng_seed=seed,
                                         disabled={"Mosaic", "MixUp", "CopyPaste"} if closed else ())
        img = sample["image"]
        h, w = img.shape[:2]
        boxes = sample["boxes"]
        # normalized xywh targets
        if boxes.shape[0]:
            xywh = np.stack([
                (boxes[:, 0] + boxes[:, 2]) / 2 / w,
                (boxes[:, 1] + boxes[:, 3]) / 2 / h,
                (boxes[:, 2] - boxes[:, 0]) / w,
                (boxes[:, 3] - boxes[:, 1]) / h,
            ], 1)
        else:
            xywh = np.zeros((0, 4), dtype=np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))  # uint8 CHW RGB
        return {
            "img": tensor,
            "cls": torch.from_numpy(sample["cls"]).float(),
            "bboxes": torch.from_numpy(xywh.astype(np.float32)),
            "path": sample["path"],
            "orig_shape": sample.get("orig_shape", (h, w)),
            "ratio": sample.get("ratio", 1.0),
            "pad": sample.get("pad", (0.0, 0.0)),
        }

    def class_counts(self, nc: int) -> np.ndarray:
        labeled = [lbl[:, 0] for lbl in self.labels if len(lbl)]
        if not labeled:
            return np.zeros(nc, dtype=np.int64)
        all_cls = np.concatenate(labeled).astype(int)
        valid = (all_cls >= 0) & (all_cls < nc)
        return np.bincount(all_cls[valid], minlength=nc).astype(np.int64)
