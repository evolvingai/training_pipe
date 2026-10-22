# Utilities — `trainer.detection.utils`

## Overview

Utility modules live in `trainer/detection/utils/` and are used throughout the framework.

| Module | Key exports | Purpose |
|---|---|---|
| `boxes.py` | `xywh2xyxy`, `xyxy2xywh`, `box_iou`, `bbox_iou`, `clip_boxes`, `scale_boxes`, `dist2bbox`, `bbox2dist`, `make_anchors` | Box format conversions and IoU variants |
| `nms.py` | `nms`, `non_max_suppression` | Non-maximum suppression |
| `ema.py` | `ModelEMA` | Exponential moving average of model weights |
| `metrics.py` | `DetMetrics`, `ConfusionMatrix`, `match_predictions`, `compute_ap` | Detection metrics (documented in [metrics.md](metrics.md)) |
| `logging.py` | `LOGGER`, `increment_path`, `format_seconds` | Logging helpers (documented in [logging.md](logging.md)) |
| `plotting.py` | `draw_boxes`, `plot_curve`, `make_grid` | Visualization helpers |
| `checks.py` | `select_device`, `autobatch`, `init_seeds` | Environment and device checks |

---

## Box utilities (`trainer/detection/utils/boxes.py`)

All functions accept both `torch.Tensor` and `np.ndarray`. Boxes are float tensors with the last dimension being 4.

### Format conversions

| Function | Input | Output | Description |
|---|---|---|---|
| `xywh2xyxy(x)` | `(..., 4)` cx cy w h | `(..., 4)` x1 y1 x2 y2 | Center+size → corner |
| `xyxy2xywh(x)` | `(..., 4)` x1 y1 x2 y2 | `(..., 4)` cx cy w h | Corner → center+size |
| `clip_boxes(boxes, shape)` | `(..., 4)` xyxy, `(h, w)` | `(..., 4)` xyxy | Clip to image boundary (in-place) |
| `scale_boxes(boxes, from_shape, to_shape, pad, gain)` | `(..., 4)` xyxy | `(..., 4)` xyxy | Map from letterboxed back to original image |

`scale_boxes` is the inverse of `LetterBox`. `from_shape` is the letterboxed size; `to_shape` is the original image size. `pad = (left, top)` and `gain` are the outputs of `LetterBox` stored in the sample.

### Anchor utilities

```python
from trainer.detection.utils.boxes import make_anchors, dist2bbox, bbox2dist

# Generate anchor center points and stride tensors from feature maps
anchors, strides = make_anchors(feats, model.stride, grid_cell_offset=0.5)
# anchors: (A, 2)  — center x, y in grid units
# strides: (A, 1)  — stride for each anchor

# Decode DFL distances to bounding boxes
boxes_xyxy = dist2bbox(dist, anchor_points, xywh=False)

# Encode bounding boxes as DFL distances
dist = bbox2dist(boxes_xyxy, anchor_points, reg_max=16)
```

`make_anchors` creates one anchor per grid cell across all feature map scales.

### IoU functions

```python
from trainer.detection.utils.boxes import box_iou, bbox_iou

# Pairwise IoU: (N, 4) x (M, 4) -> (N, M)
iou_matrix = box_iou(boxes_a, boxes_b)

# Elementwise IoU or CIoU: (..., 4) x (..., 4) -> (...)
iou = bbox_iou(pred, target, ciou=True)
```

`bbox_iou` with `ciou=True` computes **Complete IoU**:

```
CIoU = IoU - ρ²/c² - α*v
v = (4/π²) * (arctan(w_gt/h_gt) - arctan(w_pred/h_pred))²
α = v / (1 - IoU + v)
```

where `ρ²` is the squared center distance and `c²` is the squared enclosing diagonal.

---

## NMS (`trainer/detection/utils/nms.py`)

### `nms(boxes, scores, iou_thres)`

Greedy single-class NMS.

```python
from trainer.detection.utils.nms import nms

keep = nms(boxes, scores, iou_thres=0.7)
# keep: (k,) indices of retained boxes, sorted by descending score
```

### `non_max_suppression(prediction, conf_thres, iou_thres, max_det, agnostic, per_class_conf, max_nms)`

Batched multi-class NMS on raw head output.

```python
from trainer.detection.utils.nms import non_max_suppression

preds = non_max_suppression(
    prediction,       # (B, 4+nc, A) — xywh boxes + per-class sigmoid scores
    conf_thres=0.25,
    iou_thres=0.7,
    max_det=300,
    agnostic=False,   # True = class-agnostic NMS
    per_class_conf=None,  # optional (nc,) tensor overriding conf_thres per class
)
# Returns list of B tensors, each (n, 6): [x1, y1, x2, y2, conf, cls]
```

**Class-aware NMS** is implemented by the coordinate-offset trick: boxes for different classes are displaced by `cls * 7680` so they never overlap across classes.

`per_class_conf` allows different thresholds per class (set by `Predictor.set_per_class_thresholds`). When provided, `conf_thres` is ignored and each detection's confidence is compared to its class-specific threshold.

---

## ModelEMA (`trainer/detection/utils/ema.py`)

```python
from trainer.detection.utils.ema import ModelEMA

ema = ModelEMA(model, decay_max=0.9999, warmup_steps=2000)

# Called after every optimizer step:
ema.update(model)

# Use EMA weights for validation:
results = validate(ema.ema, ...)

# Save/load:
state = ema.state_dict()
ema.load_state_dict(state)
```

### Warmup-aware decay

The decay ramps up from 0 toward `decay_max` following:

```
decay(step) = decay_max * (1 - exp(-step / warmup_steps))
```

Early steps have low decay so the EMA tracks the model closely. After `warmup_steps` the decay approaches `decay_max` (very slow EMA).

For non-float parameters (e.g. `num_batches_tracked` in BN), the EMA just copies the current value.

The EMA model is kept in `.eval()` mode with `requires_grad=False` throughout.

---

## Plotting (`trainer/detection/utils/plotting.py`)

### `draw_boxes(image, boxes, cls, names, conf)`

```python
from trainer.detection.utils.plotting import draw_boxes

annotated = draw_boxes(img, boxes, cls, names={0: "cat"}, conf=conf)
```

Draws xyxy bounding boxes with labels and confidence scores on a BGR numpy image. Returns a copy.

### `plot_curve(x, ys, out_path, title)`

```python
from trainer.detection.utils.plotting import plot_curve

png = plot_curve(
    steps,
    {"train/box": box_series, "train/cls": cls_series},
    out_path="runs/exp/curve_losses.png",
    title="losses",
)
```

Renders a multi-series line chart and saves it as PNG. Returns the output `Path`. Used by `generate_report` to embed training curves.

### `make_grid(images, cols)`

```python
from trainer.detection.utils.plotting import make_grid

grid = make_grid(list_of_images, cols=4)
```

Tiles a list of same-size BGR images into a grid. Used by `viz-augment` to produce the augmentation preview.

---

## Device and environment checks (`trainer/detection/utils/checks.py`)

### `select_device(device="")`

```python
from trainer.detection.utils.checks import select_device

device = select_device("")      # auto: CUDA if available, else CPU
device = select_device("cuda:0")
device = select_device("cpu")
```

Returns a `torch.device`. Logs the selected device and, for CUDA, the GPU name and memory.

### `autobatch(model, imgsz, device, amp)`

Finds the largest power-of-two batch size that fits in GPU memory for the given model and image size. Runs a binary search with dummy forward passes.

### `init_seeds(seed, deterministic)`

Sets `random`, `numpy`, and `torch` seeds. When `deterministic=True`, also sets `torch.backends.cudnn.deterministic=True` and `benchmark=False` (at the cost of some speed).
