# Trainer — User Manual

## Overview

Trainer is a pluggable PyTorch training framework for object detection, image
classification, video/action recognition, and VLM LoRA tuning. All task
families share one registry, Python API, CLI, automation queue, VM agent, and
artifact flow. The detector backend additionally provides augmentation,
EMA/AMP training, COCO-style validation, threshold tuning, inference,
ONNX/TensorRT export, and evaluation reports.

---

## Installation

```bash
pip install -e .                      # core (torch, opencv)
pip install -e ".[track]"             # + MLflow experiment tracking
pip install -e ".[deploy]"            # + onnxruntime, tensorrt export
pip install -e ".[dev]"               # + pytest, ruff, mypy
```

**Requirements:** Python ≥ 3.11, PyTorch ≥ 2.1, OpenCV ≥ 4.8.

---

## Dataset format

Labels use the same layout as YOLO-style datasets:

```
dataset/
├── images/
│   ├── train/   *.jpg / *.png / ...
│   └── val/
└── labels/
    ├── train/   *.txt   (one per image)
    └── val/
```

Each label file has one row per object:

```
<class_id>  <cx>  <cy>  <w>  <h>
```

All coordinates are **normalized to [0, 1]** relative to image dimensions. Label files that are absent or empty mean the image has no objects.

### Dataset TOML

Create a TOML that points trainer at your data:

```toml
root = "/path/to/dataset"
nc = 3
names = {0 = "cat", 1 = "dog", 2 = "bird"}

[train]
path = "images/train"   # relative to root

[val]
path = "images/val"
```

See `trainer/detection/cfg/dataset.example.toml` for a full example.

---

## CLI

All operations go through a single entry-point:

```
trainer <command> [key=value ...]
```

### Commands

| Command | Purpose | Minimum required args |
|---|---|---|
| `train` | Train from scratch | `data=` |
| `finetune` | Transfer-learn from a checkpoint | `model=<.pt>` `data=` |
| `val` | Evaluate a checkpoint | `model=<.pt>` `data=` |
| `predict` | Run inference on images | `model=<.pt>` `source=` |
| `export` | Export to ONNX or TensorRT | `model=<.pt>` |
| `tune` | Hyperparameter evolution | `data=` |
| `tune-thresh` | Per-class confidence threshold tuning | `model=<.pt>` `data=` |
| `report` | Generate HTML evaluation report | `model=<.pt>` `data=` |
| `viz-augment` | Preview augmentation pipeline | `data=` |
| `bench-data` | Benchmark dataloader throughput | `data=` |
| `version` | Print installed version | — |

### Examples

```bash
# Train the nano model for 100 epochs
trainer train data=dataset.toml model=trainer-n epochs=100 batch=32

# Validate a checkpoint
trainer val model=runs/exp/best.pt data=dataset.toml

# Inference on a directory (saves annotated images to runs/predict/)
trainer predict model=runs/exp/best.pt source=test_images/ conf=0.4

# Fine-tune a pre-trained checkpoint on new data, freezing the first 10 layers
trainer finetune model=runs/exp/best.pt data=new_dataset.toml freeze=10

# Export to ONNX
trainer export model=runs/exp/best.pt format=onnx imgsz=640

# Export to TensorRT FP16
trainer export model=runs/exp/best.pt format=tensorrt half=true

# Tune per-class thresholds and save to thresholds.toml
trainer tune-thresh model=runs/exp/best.pt data=dataset.toml save=thresholds.toml

# Generate an HTML report (optionally including per-class thresholds)
trainer report model=runs/exp/best.pt data=dataset.toml thresholds=thresholds.toml

# Visualize augmentations for a dataset (saves augment_preview.png)
trainer viz-augment data=dataset.toml n=16

# Benchmark dataloader at current batch/worker settings
trainer bench-data data=dataset.toml n=50
```

All `key=value` args override the corresponding key in `trainer/detection/cfg/default.toml`. Boolean values accept `true`/`false`.

---

## Docker

The `docker/` directory contains a `Dockerfile` and a `docker-compose.yml` with three independent services: `train`, `eval`, and `export`. All services share the same image and mount two host directories into the container:

| Mount | Purpose |
|---|---|
| `DATA_DIR` → `/data` | datasets (read-only at runtime) |
| `RUNS_DIR` → `/runs` | training outputs, checkpoints, exports |

### Build

```bash
# run from repo root
docker compose -f docker/docker-compose.yml build
```

### Train

```bash
# defaults: trainer-n, 100 epochs, batch 16, imgsz 640
DATA_DIR=/path/to/datasets RUNS_DIR=/path/to/runs \
  docker compose -f docker/docker-compose.yml run --rm train

# override model, epochs, batch
DATA_DIR=/path/to/datasets RUNS_DIR=/path/to/runs \
DATA=/data/dataset.toml MODEL=trainer-s EPOCHS=50 BATCH=32 NAME=exp_s \
  docker compose -f docker/docker-compose.yml run --rm train
```

Available env vars for `train`:

| Variable | Default | Maps to |
|---|---|---|
| `DATA` | `/data/dataset.toml` | `data=` |
| `MODEL` | `trainer-n` | `model=` |
| `EPOCHS` | `100` | `epochs=` |
| `BATCH` | `16` | `batch=` |
| `IMGSZ` | `640` | `imgsz=` |
| `NAME` | `exp` | `name=` |

### Eval

```bash
DATA_DIR=/path/to/datasets RUNS_DIR=/path/to/runs \
CHECKPOINT=/runs/exp/best.pt DATA=/data/dataset.toml \
  docker compose -f docker/docker-compose.yml run --rm eval
```

Available env vars for `eval`:

| Variable | Default | Maps to |
|---|---|---|
| `CHECKPOINT` | `/runs/exp/best.pt` | `model=` |
| `DATA` | `/data/dataset.toml` | `data=` |
| `NAME` | `eval` | `name=` |

### Export

```bash
# ONNX (default)
RUNS_DIR=/path/to/runs CHECKPOINT=/runs/exp/best.pt \
  docker compose -f docker/docker-compose.yml run --rm export

# TensorRT FP16
RUNS_DIR=/path/to/runs CHECKPOINT=/runs/exp/best.pt FORMAT=tensorrt HALF=true \
  docker compose -f docker/docker-compose.yml run --rm export
```

Available env vars for `export`:

| Variable | Default | Maps to |
|---|---|---|
| `CHECKPOINT` | `/runs/exp/best.pt` | `model=` |
| `FORMAT` | `onnx` | `format=` |
| `HALF` | `false` | `half=` |
| `INT8` | `false` | `int8=` |

---

## Python API

```python
from trainer.detection.api import Model

# Create a model (architecture or checkpoint)
model = Model("trainer-n")              # fresh nano model
model = Model("runs/exp/best.pt")       # trained checkpoint

# Train
result = model.train("dataset.toml", epochs=100, batch=32)

# Fine-tune (auto-remaps head if nc differs)
result = model.finetune("new_dataset.toml", freeze=10, epochs=50)

# Validate
result = model.val("dataset.toml")
# result: dict with mAP50, mAP50-95, precision, recall, ap50_per_class,
#         per_image_stats (PerImageStats), confusion (np.ndarray), …

# Predict
results = model.predict("images/", conf=0.4)
for r in results:
    print(r.path, len(r), "detections")
    img = r.plot()          # numpy BGR image with boxes drawn
    txt = r.to_label_txt()  # YOLO-format label string

# Export
model.export(format="onnx", imgsz=640)
model.export(format="tensorrt", half=True)

# Tune thresholds
thresh = model.tune_thresholds("dataset.toml", save_path="thresholds.toml")
```

---

## Run directory

Training writes all outputs under `runs/<name>/` (auto-incremented to avoid collisions):

```
runs/exp/
├── config.toml      # full resolved config snapshot
├── results.csv      # per-epoch metrics (step, train/box, train/cls, train/dfl, val/mAP50, …)
├── last.pt          # latest checkpoint (resume-able)
├── best.pt          # checkpoint with best fitness metric
└── epoch<N>.pt      # periodic checkpoint (if save_period > 0)
```

To resume training from a run:

```bash
trainer train data=dataset.toml resume=true
# or point to a specific checkpoint:
trainer train data=dataset.toml resume=runs/exp2/last.pt
```

---

## Configuration reference

All keys live in `trainer/detection/cfg/default.toml` and can be overridden on the CLI or in the Python API. Key groups:

| Group | Keys |
|---|---|
| General | `task`, `model`, `data`, `project`, `name`, `seed`, `device`, `exist_ok`, `deterministic` |
| Training | `epochs`, `batch`, `imgsz`, `workers`, `optimizer`, `lr0`, `lrf`, `cos_lr`, `momentum`, `weight_decay`, `warmup_*`, `nbs`, `amp`, `freeze`, `resume`, `save_period`, `compile`, `channels_last`, `autobatch` |
| EMA | `ema_decay`, `ema_warmup_steps` |
| Early stop | `patience`, `early_stop_metric`, `early_stop_min_delta`, `early_stop_min_epochs`, `early_stop_restore_best` |
| Loss | `box`, `cls`, `dfl`, `cls_loss`, `focal_gamma`, `tal_topk`, `reg_max` |
| Augmentation | `mosaic`, `mosaic9`, `close_mosaic`, `mixup`, `mixup_alpha`, `copy_paste`, `degrees`, `translate`, `scale`, `shear`, `perspective`, `flipud`, `fliplr`, `hsv_h`, `hsv_s`, `hsv_v`, `blur`, `noise`, `jpeg`, `erasing`, `square_crop` |
| Data loader | `cache`, `rect`, `fraction`, `balanced_sampling` |
| Inference | `conf`, `iou`, `max_det`, `val_conf`, `val_period` |
| Tracking | `mlflow`, `mlflow_experiment` |
| Export | `format`, `opset`, `dynamic`, `half`, `int8`, `workspace` |
| Threshold tuning | `thresh_fitness`, `thresh_sweep_steps`, `thresh_min_fitness` |
| Hyperparam tuning | `tune_iterations`, `tune_epochs`, `tune_patience`, `tune_min_delta`, `tune_max_trials` |

---

## Model scales

Built-in models follow the naming scheme `trainer-<scale>`:

| Scale | Suffix | Depth | Width | Max channels |
|---|---|---|---|---|
| Nano | `-n` | 0.33 | 0.25 | 1024 |
| Small | `-s` | 0.33 | 0.50 | 1024 |
| Medium | `-m` | 0.67 | 0.75 | 768 |
| Large | `-l` | 1.00 | 1.00 | 512 |
| XLarge | `-x` | 1.00 | 1.25 | 512 |

---

## Inputs and outputs summary

| Operation | Input | Output |
|---|---|---|
| `train` | dataset TOML, config overrides | `runs/<name>/best.pt`, `last.pt`, `results.csv` |
| `val` | `.pt` checkpoint, dataset TOML | metrics dict (mAP50, mAP50-95, P, R, per-class AP, per-image TP/FP/FN/TN) |
| `predict` | `.pt` checkpoint, image dir/file/txt | list of `Results` (boxes, conf, cls) |
| `export` | `.pt` checkpoint | `.onnx` or `.engine` file |
| `tune-thresh` | `.pt` checkpoint, dataset TOML | `thresholds.toml`, diagnostics dict |
| `report` | `.pt` checkpoint, dataset TOML | `report.html`, `per_image_stats.csv`, `fp_detail.csv` (when FPs exist) |
