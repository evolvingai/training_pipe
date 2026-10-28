"""COCO-format detection dataset.

Annotations are loaded from a COCO JSON file (instances format).  Labels are
converted from absolute ``[x, y, w, h]`` pixel boxes to the same normalized
``[cls, cx, cy, w, h]`` format used by ``DetectionDataset``, so the rest of
the pipeline (transforms, collation, trainer) is unchanged.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from trainer.detection.data.dataset import IMG_SUFFIXES, DetectionDataset
from trainer.detection.utils.logging import LOGGER

_COCO_CACHE_VERSION = "coco2"


def parse_coco_json(annotations_json: str | Path) -> tuple[dict, dict, dict]:
    """Parse a COCO instances JSON file.

    Returns:
        images_meta  -- {image_id: {file_name, height, width}}
        image_labels -- {image_id: ndarray (N, 5) cls cx cy w h normalized}
        category_map -- {coco_category_id: 0-indexed class_id}

    iscrowd=1 annotations are skipped — crowd regions are excluded from both
    training labels and evaluation matching.  For eval-time crowd suppression
    use ``coco_sota_fixes.coco_evaluate`` which delegates to pycocotools.
    """
    data = json.loads(Path(annotations_json).read_text())

    cats = sorted(data.get("categories", []), key=lambda c: c["id"])
    category_map: dict[int, int] = {c["id"]: i for i, c in enumerate(cats)}

    images_meta: dict[int, dict] = {img["id"]: img for img in data.get("images", [])}

    image_labels: dict[int, list] = {img_id: [] for img_id in images_meta}
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        img_id = ann["image_id"]
        if img_id not in image_labels:
            continue
        cat_id = ann["category_id"]
        if cat_id not in category_map:
            continue
        meta = images_meta[img_id]
        h, w = int(meta["height"]), int(meta["width"])
        x, y, bw, bh = ann["bbox"]
        if h <= 0 or w <= 0 or not np.isfinite([x, y, bw, bh]).all() or bw <= 0 or bh <= 0:
            continue
        x1, y1 = max(float(x), 0.0), max(float(y), 0.0)
        x2, y2 = min(float(x + bw), float(w)), min(float(y + bh), float(h))
        if x2 <= x1 or y2 <= y1:
            continue
        image_labels[img_id].append([
            float(category_map[cat_id]),
            ((x1 + x2) / 2) / w,
            ((y1 + y2) / 2) / h,
            (x2 - x1) / w,
            (y2 - y1) / h,
        ])

    image_labels_np: dict[int, np.ndarray] = {
        img_id: (np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32))
        for img_id, rows in image_labels.items()
    }
    return images_meta, image_labels_np, category_map


class CocoDetectionDataset(DetectionDataset):
    """Detection dataset loaded from COCO-format JSON annotations.

    Produces identical output tensors to ``DetectionDataset`` so it is
    fully compatible with the existing augmentation, collation, and training
    pipeline.

    Args:
        images_dir:       Directory containing the image files listed in the
                          COCO JSON (``file_name`` values are resolved relative
                          to this directory).
        annotations_json: Path to the COCO instances JSON file.
        imgsz:            Target image size (long side) for pre-resize caching.
        transforms:       Augmentation pipeline (same interface as for
                          ``DetectionDataset``).
        cache:            ``"none"`` | ``"ram"`` | ``"disk"``
        fraction:         Fraction of images to use (applied before scanning).
    """

    def __init__(
        self,
        images_dir: str | Path,
        annotations_json: str | Path,
        imgsz: int = 640,
        transforms=None,
        cache: str = "none",
        fraction: float = 1.0,
    ) -> None:
        self._coco_json = Path(annotations_json)
        images_dir = Path(images_dir)

        images_meta, image_labels_np, self.category_map = parse_coco_json(self._coco_json)
        self.annotations_json = self._coco_json.resolve()
        self.category_id_for_class = {class_id: category_id for category_id, class_id in self.category_map.items()}
        self.image_id_by_path: dict[str, int] = {}
        self._coco_shapes = {
            str((images_dir / meta["file_name"]).resolve()): (int(meta["height"]), int(meta["width"]))
            for meta in images_meta.values()
        }

        pairs: list[tuple[str, np.ndarray]] = []
        for img_id, meta in images_meta.items():
            p = images_dir / meta["file_name"]
            if p.suffix.lower() not in IMG_SUFFIXES:
                continue
            pairs.append((str(p), image_labels_np[img_id]))
            self.image_id_by_path[str(p)] = int(img_id)
            self.image_id_by_path[str(p.resolve())] = int(img_id)

        if fraction < 1.0:
            pairs = pairs[: max(1, int(len(pairs) * fraction))]

        if not pairs:
            raise FileNotFoundError(f"no images found in COCO dataset: {images_dir}")

        # needed by _load_or_build_cache before labels are set
        self._coco_pairs = pairs

        # Bypass DetectionDataset.__init__ (which expects YOLO label files) and
        # call Dataset.__init__ directly, then replicate the setup manually.
        Dataset.__init__(self)
        self.imgsz = imgsz
        self.nc = len(self.category_map)
        self.transforms = transforms
        self.cache_mode = cache

        img_paths = [p for p, _ in pairs]
        self.paths, self.shapes, self.labels = self._load_or_build_cache(img_paths)
        self._ram: list[np.ndarray | None] = [None] * len(self.paths)
        self.skipped_total = len(pairs) - len(self.paths)

        if self.cache_mode == "disk":
            self._build_disk_cache()
        if self.cache_mode == "ram":
            self._preload_ram_cache()
        if transforms is not None and hasattr(transforms, "set_dataset"):
            transforms.set_dataset(self)

    # -- cache (COCO-specific key + validation) --------------------------------

    def _cache_key(self, img_paths: list[str]) -> str:  # type: ignore[override]
        def _stat_one(p: str) -> str:
            try:
                s = Path(p).stat()
                return f"{p}:{s.st_mtime_ns}:{s.st_size}"
            except FileNotFoundError:
                return p

        hasher = hashlib.sha256()
        hasher.update(_COCO_CACHE_VERSION.encode())
        ann_stat = self._coco_json.stat()
        hasher.update(f"{self._coco_json}:{ann_stat.st_mtime_ns}:{ann_stat.st_size}".encode())
        with cf.ThreadPoolExecutor() as pool:
            for entry in pool.map(_stat_one, img_paths):
                hasher.update(entry.encode())
        return hasher.hexdigest()[:16]

    def _load_or_build_cache(self, img_paths: list[str]):  # type: ignore[override]
        cache_path = (
            Path(img_paths[0]).parent
            / f".trainer_coco_cache_{self._cache_key(img_paths)}.npz"
        )
        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    paths = [str(p) for p in data["paths"]]
                    shapes = data["shapes"]
                    flat, splits = data["labels_flat"], data["labels_splits"]
                    labels = [flat[a:b].reshape(-1, 5) for a, b in zip(splits[:-1], splits[1:])]
                LOGGER.info(f"COCO dataset cache hit: {len(paths)} images ({cache_path.name})")
                return paths, shapes, labels
            except (OSError, ValueError, KeyError) as error:
                LOGGER.warning(f"ignoring invalid COCO dataset cache {cache_path}: {error}")

        LOGGER.info(f"scanning COCO dataset ({len(img_paths)} images)...")
        lbl_by_path = {p: lbl for p, lbl in self._coco_pairs}
        paths: list[str] = []
        shapes: list[tuple[int, int]] = []
        labels: list[np.ndarray] = []
        skipped = 0

        def _validate(img_path: str):
            img = cv2.imread(img_path)  # channel order is irrelevant here: only the shape is used
            if img is None:
                return img_path, None, None, "unreadable image"
            expected_shape = self._coco_shapes.get(str(Path(img_path).resolve()))
            if expected_shape is not None and img.shape[:2] != expected_shape:
                return img_path, None, None, "image dimensions differ from COCO metadata"
            return img_path, img.shape[:2], lbl_by_path[img_path], None

        with cf.ThreadPoolExecutor() as pool:
            for img_path, shape, lbl, err in pool.map(_validate, img_paths):
                if err:
                    LOGGER.warning(f"skipping {img_path}: {err}")
                    skipped += 1
                    continue
                paths.append(img_path)
                shapes.append(shape)
                labels.append(lbl)

        if not paths:
            raise RuntimeError("all COCO dataset samples were invalid")
        if skipped:
            LOGGER.warning(f"COCO dataset scan skipped {skipped} invalid samples")

        shapes_arr = np.array(shapes, dtype=np.int32)
        flat = np.concatenate([lb.reshape(-1) for lb in labels]) if labels else np.zeros(0, np.float32)
        splits_arr = np.cumsum([0] + [lb.size for lb in labels])
        try:
            np.savez(
                cache_path,
                paths=np.array(paths),
                shapes=shapes_arr,
                labels_flat=flat.astype(np.float32),
                labels_splits=splits_arr.astype(np.int64),
            )
        except OSError as e:
            LOGGER.warning(f"could not write COCO dataset cache: {e}")
        return paths, shapes_arr, labels
