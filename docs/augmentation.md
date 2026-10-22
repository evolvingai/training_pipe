# Shared augmentation — `trainer.augmentation`

## Overview

`trainer/augmentation/` is the task-neutral augmentation subsystem used by
detection, image classification, video/action classification, and VLM tuning.
The old `trainer.detection.augment` namespace remains as a compatibility alias.
All transforms share the same calling convention and are composed into a pipeline.

Files:
- `config.py` — common defaults, external TOML loading, and reusable CLI arguments.
- `registry.py` — `Transform` base class, registry, sequence application, and pipeline builder.
- `geometric.py` — Spatial transforms that update bounding boxes.
- `photometric.py` — Pixel-level transforms that leave boxes untouched.

---

## Transform interface

Every transform is a callable:

```python
sample_out = transform(sample, rng)
```

| Argument | Type | Description |
|---|---|---|
| `sample` | `dict` | Contains `"image"` (numpy HWC BGR uint8). Detection adds `"boxes"` and `"cls"`; classification adds a soft `"target"` vector. |
| `rng` | `random.Random` | Seeded RNG for reproducibility |

Returns a modified copy of `sample` (some transforms modify in-place; the contract is that the returned dict is valid).

---

## Pipeline composition

```python
from trainer.augmentation import build_pipeline

pipeline = build_pipeline(cfg, train=True, seed=42)
sample = pipeline({"image": img, "boxes": boxes, "cls": cls})
```

`build_pipeline` reads augmentation probability keys from `cfg` and assembles an ordered list of transforms. Dataset-aware transforms (`Mosaic`, `MixUp`, `CopyPaste`) receive a reference to the dataset via `pipeline.set_dataset(dataset)`.

For temporally or spatially aligned inputs, `pipeline.apply_many(samples)`
replays the same random decisions across every image. Video classification uses
this so all frames in a clip receive identical geometry.

An external profile can be shared by any training method:

```toml
[augmentation]
fliplr = 0.5
degrees = 10.0
mixup = 0.2
blur = 0.1
jpeg = 0.1
```

```bash
trainer train task=classification data=/data/species \
  augmentation_config=/data/augmentations/robust.toml
```

Explicit command options override profile values.

---

## Geometric transforms (`trainer/augmentation/geometric.py`)

### `LetterBox`

Resize preserving aspect ratio and pad to a square canvas.

```python
LetterBox(imgsz=640, pad_value=114, center=True)
```

- Scales so the longer edge fits `imgsz`.
- Pads with constant color (default 114, neutral grey).
- Stores `ratio` and `pad=(left, top)` in the sample for coordinate back-mapping during inference.
- Boxes are scaled and offset to match the new canvas.

### `RandomFlip`

```python
RandomFlip(direction="horizontal", p=0.5)
```

Flips image horizontally or vertically with probability `p`. Updates box coordinates accordingly. Config key: `fliplr` (horizontal), `flipud` (vertical).

### `RandomPerspective`

```python
RandomPerspective(degrees=0, translate=0.1, scale=0.5, shear=0, perspective=0, pad_value=114, min_visibility=0.1)
```

Applies a random affine (or perspective) warp. Composition order: center → perspective → rotation+scale → shear → translation → un-center.

All four corners of each box are transformed, then a new axis-aligned bounding box is fitted. Boxes whose visible area falls below `min_visibility * original_area` are filtered out.

Config keys: `degrees`, `translate`, `scale`, `shear`, `perspective`.

### `SquareCrop`

```python
SquareCrop(jitter=0.0, min_visibility=0.1)
```

Crops the largest possible square from a non-square image. `jitter` adds random offset along the long axis (in units of `max_offset`). Enabled by config key `square_crop=true`, `square_crop_jitter`.

### `Mosaic`

```python
Mosaic(imgsz=640, p=1.0, p9=0.0, pad_value=114)
```

Combines 4 (or 9, with probability `p9`) randomly-sampled images into a mosaic. Each tile is resized to fit within the grid, placed at a random pivot point, and boxes are offset accordingly. Requires dataset access (`requires_dataset = True`).

For classification, visible tile areas weight the resulting soft class target.

Config keys: `mosaic` (4-tile probability), `mosaic9` (9-tile extra probability).

**4-tile layout**: a random center point `(cx, cy)` is chosen within `[0.5*imgsz, 1.5*imgsz]`. The main image goes to the top-left quadrant. The canvas is `2*imgsz × 2*imgsz` before being cropped downstream by `RandomPerspective`.

**9-tile layout**: tiles are arranged in a 3×3 grid, then a `2*imgsz` center crop is taken.

### `MixUp`

```python
MixUp(p=0.0, alpha=1.5)
```

Blends two images pixel-wise with `lambda ~ Beta(alpha, alpha)`. Detection label
sets are unioned; classification targets are mixed by the same lambda. Requires dataset access.

Config keys: `mixup` (probability), `mixup_alpha`.

### `CopyPaste`

```python
CopyPaste(p=0.0, iou_range=(0.10, 0.55), max_paste=3, attempts=10)
```

Copies object crops from a randomly-sampled donor image and pastes them onto the current image. A paste is accepted only when the IoU of the new box with existing boxes falls within `iou_range` (occlusion simulation). Up to `max_paste` objects are pasted per image, with `attempts` random placement tries each.

When boxes are unavailable, classification uses a CutMix-style donor region and
weights the soft target by the actual pasted area.

Config keys: `copy_paste` (probability), `copy_paste_iou`.

---

## Photometric transforms (`trainer/augmentation/photometric.py`)

All photometric transforms leave `boxes` and `cls` untouched.

### `RandomHSV`

```python
RandomHSV(h=0.015, s=0.7, v=0.4, p=1.0)
```

Jitters hue, saturation, and value via per-channel LUT lookup. Config keys: `hsv_h`, `hsv_s`, `hsv_v`.

### `BrightnessContrast`

```python
BrightnessContrast(brightness=0.2, contrast=0.2, p=0.5)
```

Applies `cv2.convertScaleAbs(alpha=1+Δcontrast, beta=255*Δbrightness)`.

### `MotionBlur`

```python
MotionBlur(p=0.0, max_kernel=9)
```

Applies a horizontal or vertical motion blur kernel of random odd size in `[3, max_kernel]`. Config key: `blur`.

### `GaussNoise`

```python
GaussNoise(p=0.0, sigma_max=12.0)
```

Adds Gaussian noise with `sigma ~ Uniform(2, sigma_max)`. Config key: `noise`.

### `JPEGArtifacts`

```python
JPEGArtifacts(p=0.0, quality_range=(30, 80))
```

Encodes and decodes the image as JPEG at a random quality in `quality_range`. Config key: `jpeg`.

### `RandomErasing`

```python
RandomErasing(p=0.0, area_range=(0.02, 0.1), max_patches=3)
```

Fills random rectangular patches with uniform noise (occlusion simulation). Config key: `erasing`.

---

## `filter_boxes` helper

```python
from trainer.augmentation import filter_boxes

sample = filter_boxes(sample, original_areas=None, min_visibility=0.1)
```

Clips boxes to the image boundary and removes:

- Boxes with zero or negative area after clipping.
- Boxes whose clipped area is less than `min_visibility * original_area` (if `original_areas` is provided).

Called internally by `RandomPerspective`, `SquareCrop`, and `Mosaic`.

---

## Augmentation config reference

| Key | Default | Transform |
|---|---|---|
| `mosaic` | 1.0 | `Mosaic` probability |
| `mosaic9` | 0.0 | 9-tile mosaic extra probability |
| `close_mosaic` | 10 | Disable mosaic for last N epochs |
| `mixup` | 0.0 | `MixUp` probability |
| `mixup_alpha` | 1.5 | Beta distribution alpha for MixUp |
| `copy_paste` | 0.0 | `CopyPaste` probability |
| `copy_paste_iou` | [0.10, 0.55] | IoU range for paste acceptance |
| `degrees` | 0.0 | Rotation range (±degrees) |
| `translate` | 0.1 | Translation fraction |
| `scale` | 0.5 | Scale jitter (±fraction) |
| `shear` | 0.0 | Shear angle (±degrees) |
| `perspective` | 0.0 | Perspective distortion |
| `flipud` | 0.0 | Vertical flip probability |
| `fliplr` | 0.5 | Horizontal flip probability |
| `hsv_h` | 0.015 | Hue jitter |
| `hsv_s` | 0.7 | Saturation jitter |
| `hsv_v` | 0.4 | Value jitter |
| `brightness_contrast` | 0.0 | Brightness/contrast probability |
| `brightness` | 0.2 | Brightness magnitude |
| `contrast` | 0.2 | Contrast magnitude |
| `blur` | 0.0 | Motion blur probability |
| `noise` | 0.0 | Gaussian noise probability |
| `jpeg` | 0.0 | JPEG artifact probability |
| `erasing` | 0.0 | Random erasing probability |
| `square_crop` | `false` | Enable SquareCrop |
| `square_crop_jitter` | 0.0 | SquareCrop long-axis jitter fraction |
| `min_box_visibility` | 0.1 | Min visibility fraction after geometric transforms |

## Task behavior

| Task | Shared transforms | Multi-image transforms | Target behavior |
|---|---|---|---|
| Detection | All | Mosaic, MixUp, CopyPaste | Bounding boxes and class rows are updated |
| Image classification | All | Mosaic, MixUp, CopyPaste/CutMix | Soft class targets are mixed correctly |
| Video/action | Image-only transforms | Disabled | One random transform is replayed across the clip |
| VLM tuning | Image-only transforms | Disabled | Image/prompt pairing is preserved |
