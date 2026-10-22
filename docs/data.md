# Data — `trainer.detection.data`

## Overview

`trainer/detection/data/` handles dataset loading, label caching, and dataloader construction.

Files:
- `dataset.py` — `DetectionDataset`: `torch.utils.data.Dataset` with binary cache.
- `build.py` — `build_dataloader`, `parse_dataset_toml`, split resolution.

---

## Dataset format

Images and label files must follow the YOLO directory layout:

```
root/
├── images/
│   ├── train/   *.jpg, *.png, *.bmp, *.webp, *.tif, *.tiff
│   └── val/
└── labels/
    ├── train/   *.txt
    └── val/
```

Label path is derived from the image path by replacing the last `images` path component with `labels` and the suffix with `.txt`.

Each label file has one row per object:

```
<class_id>  <cx_norm>  <cy_norm>  <w_norm>  <h_norm>
```

Coordinates are normalized to `[0, 1]`. A missing or empty label file means no objects.

---

## `DetectionDataset` (`trainer/detection/data/dataset.py`)

```python
from trainer.detection.data.dataset import DetectionDataset

ds = DetectionDataset(
    img_paths=["path/to/img1.jpg", ...],
    imgsz=640,
    transforms=pipeline,     # augmentation pipeline
    cache="none",            # "none" | "ram" | "disk"
    fraction=1.0,            # load only this fraction of images
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `img_paths` | `list[str]` | — | Absolute or relative image paths |
| `imgsz` | `int` | 640 | Pre-resize long edge to this size before applying transforms |
| `transforms` | callable | `None` | Augmentation pipeline |
| `cache` | `str` | `"none"` | Image caching mode |
| `fraction` | `float` | 1.0 | Subsample the dataset to this fraction (first N images) |

### Label cache

On first use, the dataset scans all images and labels in parallel (via `ThreadPoolExecutor`) and writes a compressed NumPy archive (`.trainer_cache_<hash>.npz`) next to the first image. The cache key is a SHA-256 hash of all image/label file paths, mtimes, and sizes.

On subsequent runs, if the cache file exists and the hash matches, scanning is skipped.

```
.trainer_cache_a1b2c3d4e5f6g7h8.npz
```

If the cache directory is read-only, the scan result is used without saving.

### Image caching modes

| `cache` value | Behaviour |
|---|---|
| `"none"` (default) | Read from disk on every `__getitem__` call |
| `"ram"` | Decode each image once and keep it in a list; fast, high memory |
| `"disk"` | Pre-decode and save resized images as `.trainer.npy` next to originals; fast I/O |

Pre-decode resizes the long edge to `imgsz` but does not letterbox (that happens in the transform pipeline). This reduces disk I/O and is especially useful on network storage.

### `load_sample(idx)`

Returns a raw sample dict without transforms:

```python
{
    "image": np.ndarray,     # HWC BGR uint8, long edge == imgsz
    "boxes": np.ndarray,     # (n, 4) xyxy float32, in pre-resized pixel coords
    "cls":   np.ndarray,     # (n,) float32
    "path":  str,
    "orig_shape": (h, w),    # original image dimensions
}
```

### `__getitem__(idx)`

Returns a training-ready batch element:

```python
{
    "img":        torch.Tensor,  # (3, H, W) uint8 CHW BGR
    "cls":        torch.Tensor,  # (n,) float32
    "bboxes":     torch.Tensor,  # (n, 4) normalized xywh float32
    "path":       str,
    "orig_shape": (h, w),
    "ratio":      float,         # letterbox scale ratio
    "pad":        (left, top),   # letterbox padding
}
```

### `class_counts(nc)`

Returns a `(nc,)` int64 array of per-class annotation counts across the full dataset. Used by balanced sampling.

---

## `parse_dataset_toml(path)` (`trainer/detection/data/build.py`)

```python
from trainer.detection.data import parse_dataset_toml

parsed = parse_dataset_toml("dataset.toml")
# {
#   "root": Path,
#   "nc": int,
#   "names": {0: "cat", 1: "dog"},
#   "train": {"path": ..., ...},
#   "val": {"path": ..., ...},
# }
```

Reads the dataset TOML and resolves `root` relative to the TOML file's directory.

---

## `build_dataloader(data, cfg, split, batch=None)` (`trainer/detection/data/build.py`)

```python
from trainer.detection.data import build_dataloader, parse_dataset_toml

parsed = parse_dataset_toml("dataset.toml")
loader = build_dataloader(parsed, cfg, "train")
```

| Parameter | Type | Description |
|---|---|---|
| `data` | `dict` | Output of `parse_dataset_toml` |
| `cfg` | `dict` | Full config dict |
| `split` | `str` | `"train"` or `"val"` |
| `batch` | `int \| None` | Override batch size (default: `cfg["batch"]`) |

Builds a `DetectionDataset` and wraps it in a `DataLoader` with:

- `shuffle=True` for train, `False` for val.
- `num_workers=cfg["workers"]` (default 8).
- `pin_memory=True` on CUDA.
- Custom collate that stacks images and concatenates variable-length label tensors, adding a `batch_idx` column so each annotation can be tied back to its image.

### Balanced sampling

When `cfg["balanced_sampling"] = true` (train split only), a `WeightedRandomSampler` is used so that each class is approximately equally likely to appear in each batch. Weights are computed from inverse class frequency.

---

## Collate batch format

The `DataLoader` yields batches as:

```python
{
    "img":       torch.Tensor,  # (B, 3, H, W) uint8
    "cls":       torch.Tensor,  # (total_objects,) float32
    "bboxes":    torch.Tensor,  # (total_objects, 4) normalized xywh
    "batch_idx": torch.Tensor,  # (total_objects,) int — image index within batch
    "path":      list[str],
    "orig_shape": list[tuple],
}
```

The `batch_idx` tensor is used by the loss function and validator to associate each annotation with its image.
