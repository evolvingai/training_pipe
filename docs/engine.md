# Engine — `trainer.detection.engine`

## Overview

`trainer/detection/engine/` contains the four operational engines beyond training:

| Module | Class / function | Purpose |
|---|---|---|
| `validator.py` | `validate()` | COCO-protocol validation on a dataloader |
| `predictor.py` | `Predictor`, `Results` | Inference pipeline with coordinate back-mapping |
| `exporter.py` | `export_onnx()`, `export_tensorrt()`, `verify_onnx_parity()` | Model export |
| `thresh_tuner.py` | `tune_thresholds()`, `collect_stats()` | Per-class confidence threshold optimization |

---

## Validator (`trainer/detection/engine/validator.py`)

### `validate(model, dataloader, cfg, names, device, compute_confusion, compute_per_image, half)`

```python
from trainer.detection.engine.validator import validate

result = validate(model, val_loader, cfg, names={0: "cat"},
                  compute_confusion=True, compute_per_image=True)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | — | Eval-mode model (typically EMA) |
| `dataloader` | `DataLoader` | — | Validation dataloader |
| `cfg` | `dict` | — | Config dict (uses `val_conf`, `iou`, `max_det`) |
| `names` | `dict[int, str]` | — | Class index → name |
| `device` | `torch.device \| None` | from model | Inference device |
| `compute_confusion` | `bool` | `False` | Build a `(nc+1)×(nc+1)` confusion matrix |
| `compute_per_image` | `bool` | `False` | Track per-image per-class TP/FP/FN/TN via `PerImageStats` |
| `half` | `bool` | `False` | Cast images to FP16 |

**Returns** `result` dict (see `DetMetrics.compute()`) plus:
- `val_loss: float` — placeholder; always 0.0 (loss is not recomputed during val).
- `confusion: np.ndarray (nc+1, nc+1)` — only present when `compute_confusion=True`.
- `per_image_stats: PerImageStats` — only present when `compute_per_image=True`; see `trainer.detection.utils.metrics.PerImageStats`.

NMS parameters used:
- `val_conf` (default 0.001) — very low threshold to collect all candidates.
- `iou` (default 0.7) — NMS IoU threshold.
- `max_det` (default 300) — max detections per image.

### Helper functions

```python
from trainer.detection.engine.validator import per_class_table, log_per_class_metrics, print_val_summary

# Formatted console table
print(per_class_table(result, names))

# Log per-class AP to trackers
log_per_class_metrics(tracker, result, names, step=epoch)
```

`per_class_table` produces fixed-width text output:

```
class                AP50  AP50-95
cat                 0.823    0.612
dog                 0.741    0.534
all                 0.782    0.573
```

`log_per_class_metrics` logs metrics in the form `val/AP50_<name>` and `val/AP50-95_<name>` (spaces in class names replaced with `_`).

---

## Predictor (`trainer/detection/engine/predictor.py`)

### `Results` dataclass

```python
from trainer.detection.engine.predictor import Results

r: Results
r.path       # str — source image path
r.orig_shape # (h, w) — original image dimensions
r.boxes      # np.ndarray (n, 4) xyxy in original pixel coords
r.conf       # np.ndarray (n,)
r.cls        # np.ndarray (n,) int
r.names      # dict[int, str]
r.embedding  # np.ndarray (C,) or None — global-avg-pool of finest feature map

len(r)           # number of detections
r.plot(image)    # draw boxes on image; returns numpy BGR array
r.to_label_txt() # YOLO-format label string (normalized xywh)
r.summary()      # {"path": ..., "n": ..., "max_conf": ..., "mean_conf": ...}
```

### `Predictor`

```python
from trainer.detection.engine.predictor import Predictor

predictor = Predictor(model, cfg, names={0: "cat"}, device=device)

# Optional per-class thresholds
predictor.set_per_class_thresholds({"cat": 0.6, "dog": 0.4})

results = predictor.predict("images/", conf=0.5, embed=False, batch_size=16)
```

**`predict(source, conf, embed, batch_size)`**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | `str \| Path \| list` | — | Image path, directory, `.txt` file of paths, or list |
| `conf` | `float \| None` | from `cfg["conf"]` | Confidence threshold (overrides per-class thresholds if set) |
| `embed` | `bool` | `False` | Return embedding (global-avg-pool of finest pyramid level) |
| `batch_size` | `int` | 16 | Number of images processed per forward pass |

**Preprocessing**: images are letterboxed to `cfg["imgsz"]` (preserving aspect ratio) using `LetterBox`.

**Postprocessing**: NMS output boxes are mapped back to original image coordinates via `scale_boxes` using the stored `ratio` and `pad` from letterboxing.

**Source resolution** (`gather_sources`):
- Directory → recursively finds all files with supported image suffixes.
- `.txt` file → reads paths line-by-line.
- Single path → returns as-is.
- List → recurses on each element.

---

## Exporter (`trainer/detection/engine/exporter.py`)

### `export_onnx(model, imgsz, opset, dynamic, half, out_path, device)`

```python
from trainer.detection.engine.exporter import export_onnx

onnx_path = export_onnx(model, imgsz=640, opset=17, dynamic=False, out_path="model.onnx")
```

- Wraps the model in `_ExportWrapper` which strips the feature-map output and returns only the decoded `(B, 4+nc, A)` tensor.
- Fuses Conv+BN (`model.fuse()`) before tracing.
- Input node: `"images"`, output node: `"output"`.
- `dynamic=True` makes batch dimension dynamic.
- Logs file size on completion.

### `verify_onnx_parity(model, onnx_path, imgsz, atol, n_samples, seed)`

```python
from trainer.detection.engine.exporter import verify_onnx_parity

max_diff = verify_onnx_parity(model, "model.onnx", imgsz=640)
```

Runs `n_samples` (default 4) random inputs through both the PyTorch model and the ONNX Runtime session. Raises `RuntimeError` when `max_abs_diff > atol` (default `1e-4` for FP32). Requires `onnxruntime` (`[deploy]` extra).

Parity tolerances:

| Precision | `atol` |
|---|---|
| FP32 | 1e-4 |
| FP16 | 5e-3 |

### `export_tensorrt(onnx_path, half, int8, workspace_gb, calib_images, imgsz, out_path)`

```python
from trainer.detection.engine.exporter import export_tensorrt

engine_path = export_tensorrt("model.onnx", half=True, workspace_gb=4)
```

Builds a TensorRT engine from an ONNX file. Requires `tensorrt` and a CUDA GPU ([`deploy`] extra).

| Parameter | Default | Description |
|---|---|---|
| `half` | `False` | Enable FP16 (if platform supports it) |
| `int8` | `False` | Enable INT8 (if platform supports it) |
| `workspace_gb` | 4 | TensorRT builder workspace in GB |
| `calib_images` | `None` | List of image paths for INT8 entropy calibration |

---

## Threshold tuner (`trainer/detection/engine/thresh_tuner.py`)

### `collect_stats(model, dataloader, cfg, nc, device)`

Runs inference at a very low threshold (0.001) to collect all raw predictions matched against GT at IoU=0.5. Returns a list of per-image dicts with `conf`, `cls`, `tp`, `gt_cls` arrays.

### `tune_thresholds(stats, nc, names, cfg)`

```python
from trainer.detection.engine.thresh_tuner import collect_stats, tune_thresholds, save_thresholds

stats = collect_stats(model, val_loader, cfg, nc=3)
result = tune_thresholds(stats, nc=3, names={0: "cat", 1: "dog", 2: "bird"}, cfg=cfg)

# result = {
#   "thresholds": {"cat": 0.52, "dog": 0.38, "bird": 0.61},
#   "diagnostics": {
#     "cat":  {"status": "ok", "threshold": 0.52, "best_fitness": 0.84, "precision": 0.91, "recall": 0.78, "n_gt": 120},
#     "dog":  {"status": "ok", ...},
#     "bird": {"status": "poorly_separated", "best_fitness": 0.21, ...},
#   }
# }
```

**Algorithm**: for each class, sweeps `thresh_sweep_steps` (default 200) candidate thresholds from 0.01 to 0.99. At each threshold computes precision and recall vectorized, then evaluates the fitness function.

**Fitness functions** (config key `thresh_fitness`):

| Value | Objective |
|---|---|
| `"f1"` (default) | Maximize F1 score |
| `"precision>=<floor>"` | Maximize recall subject to precision ≥ floor |
| `"recall>=<floor>"` | Maximize precision subject to recall ≥ floor |

A class is flagged `"poorly_separated"` when the best fitness never reaches `thresh_min_fitness` (default 0.3).

### `save_thresholds(result, path)`

Writes `result["thresholds"]` to a TOML file:

```toml
[thresholds]
cat = 0.52
dog = 0.38
bird = 0.61
```

This file can be passed back at inference time:

```bash
trainer predict model=best.pt source=images/ thresholds=thresholds.toml
```

### Config keys

| Key | Default | Description |
|---|---|---|
| `thresh_fitness` | `"f1"` | Fitness function for threshold selection |
| `thresh_sweep_steps` | 200 | Number of threshold candidates to evaluate |
| `thresh_min_fitness` | 0.3 | Minimum fitness for `"ok"` status |
