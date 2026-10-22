# Metrics — `trainer.detection.utils.metrics`

## Overview

`trainer/detection/utils/metrics.py` implements COCO-protocol detection metrics without depending on `pycocotools`. It includes:

- `match_predictions` — TP matrix for one image.
- `compute_ap` — 101-point interpolated AP.
- `DetMetrics` — accumulator for dataset-wide mAP.
- `ConfusionMatrix` — confusion matrix at a fixed conf/IoU threshold.

---

## IoU thresholds

```python
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
# [0.50, 0.55, 0.60, ..., 0.95]
```

AP is computed at each threshold and averaged to form AP50-95.

---

## `match_predictions(pred_boxes, pred_cls, gt_boxes, gt_cls)`

```python
from trainer.detection.utils.metrics import match_predictions

tp = match_predictions(pred_boxes, pred_cls, gt_boxes, gt_cls)
# Returns np.ndarray of shape (n_pred, 10) dtype bool
# tp[i, t] == True means prediction i is a TP at IOU_THRESHOLDS[t]
```

**Inputs** (all for a single image):

| Parameter | Type | Shape | Description |
|---|---|---|---|
| `pred_boxes` | `torch.Tensor` | `(n_pred, 4)` | Predicted boxes in `xyxy` pixel coords, sorted by descending confidence |
| `pred_cls` | `torch.Tensor` | `(n_pred,)` | Predicted class indices |
| `gt_boxes` | `torch.Tensor` | `(n_gt, 4)` | Ground-truth boxes in `xyxy` pixel coords |
| `gt_cls` | `torch.Tensor` | `(n_gt,)` | Ground-truth class indices |

**Algorithm**: for each IoU threshold, collect all (gt, pred) pairs with IoU ≥ threshold and matching class, sort by descending IoU, then greedily match so each GT and each prediction is used at most once.

---

## `compute_ap(recall, precision)`

```python
from trainer.detection.utils.metrics import compute_ap

ap = compute_ap(recall_curve, precision_curve)
# Returns float in [0, 1]
```

101-point interpolated AP: interpolates the precision-recall curve at 101 evenly-spaced recall points and takes the trapezoidal integral. Boundary conditions: recall is padded with `[0, r[-1]+0.01]` and precision with `[1.0, 0.0]`.

---

## `DetMetrics` (`trainer/detection/utils/metrics.py`)

Accumulates per-image TP matches across the full dataset, then computes final metrics.

```python
from trainer.detection.utils.metrics import DetMetrics

metrics = DetMetrics(nc=80, names={0: "cat", 1: "dog", ...})

# Call once per image during validation:
metrics.update(pred, gt_boxes, gt_cls)

# After all images:
result = metrics.compute()
```

### `update(pred, gt_boxes, gt_cls)`

| Parameter | Type | Shape | Description |
|---|---|---|---|
| `pred` | `torch.Tensor` | `(n, 6)` | `[x1, y1, x2, y2, conf, cls]` per detection |
| `gt_boxes` | `torch.Tensor` | `(m, 4)` | Ground-truth boxes `xyxy` |
| `gt_cls` | `torch.Tensor` | `(m,)` | Ground-truth class indices |

Internally sorts predictions by descending confidence before calling `match_predictions`.

### `compute()` return dict

| Key | Type | Description |
|---|---|---|
| `mAP50` | `float` | Mean AP at IoU=0.50 across seen classes |
| `mAP50-95` | `float` | Mean AP across IoU 0.50:0.05:0.95 across seen classes |
| `precision` | `float` | Mean precision at the max-F1 operating point (IoU=0.50), across seen classes |
| `recall` | `float` | Mean recall at the max-F1 operating point (IoU=0.50), across seen classes |
| `ap50_per_class` | `np.ndarray (nc,)` | AP50 for each class (0 for unseen classes) |
| `ap_per_class` | `np.ndarray (nc,)` | AP50-95 for each class (0 for unseen classes) |
| `classes_seen` | `np.ndarray (nc,) bool` | True for classes that had at least one GT annotation |

Returns an all-zeros dict when called with no accumulated data.

---

## `ConfusionMatrix` (`trainer/detection/utils/metrics.py`)

Computes a `(nc+1) × (nc+1)` confusion matrix at fixed confidence and IoU thresholds. The last index (`nc`) represents "background" (i.e. missed GTs or false positives).

```python
from trainer.detection.utils.metrics import ConfusionMatrix

cm = ConfusionMatrix(nc=3, conf=0.25, iou_thres=0.45)

# Call once per image:
cm.update(pred, gt_boxes, gt_cls)

# Access the matrix:
matrix = cm.matrix  # np.ndarray shape (nc+1, nc+1) int64
```

### Matrix layout

```
          actual_class_0  actual_class_1  ...  background (missed)
predicted_class_0
predicted_class_1
...
background (FP)
```

- `matrix[p, g]` — prediction class `p` matched to GT class `g`.
- `matrix[nc, g]` — GT class `g` was not matched (false negative / missed detection).
- `matrix[p, nc]` — prediction class `p` had no matching GT (false positive).

### `update(pred, gt_boxes, gt_cls)`

Filters predictions below `self.conf`, then matches remaining predictions to GTs using `box_iou` ≥ `self.iou_thres`. Unmatched GTs → background column; unmatched predictions → background row.

---

---

## `PerImageStats` (`trainer/detection/utils/metrics.py`)

Tracks per-image, per-class TP / FP / FN / TN counts and records the ground-truth class at every false-positive location, enabling confusion-level diagnosis for individual images.

```python
from trainer.detection.utils.metrics import PerImageStats

stats = PerImageStats(nc=3, names={0: "cat", 1: "dog", 2: "bird"}, iou_thres=0.5)

# Call once per image during validation:
stats.update(image_id="path/to/image.jpg", pred=pred, gt_boxes=gt_boxes, gt_cls=gt_cls)

# Retrieve rows for reporting:
table_rows = stats.per_image_class_rows()   # list[list] — one row per active (image, class) pair
fp_rows    = stats.fp_detail_rows()         # list[list] — one row per FP event
```

### Matching algorithm

Unlike `match_predictions` (which masks IoU by class before matching), `PerImageStats.update` matches predicted boxes to GT boxes **without a class constraint**. This means a predicted box that overlaps a GT box of a different class is recorded as a class-confused FP rather than a background FP.

- Predictions are sorted by descending confidence before greedy matching.
- IoU threshold is configurable via `iou_thres` (default `0.5`).

### `update(image_id, pred, gt_boxes, gt_cls)`

| Parameter | Type | Shape | Description |
|---|---|---|---|
| `image_id` | `str` | — | Full image path (filename is extracted for display) |
| `pred` | `torch.Tensor` | `(n, 6)` | `[x1, y1, x2, y2, conf, cls]`, any confidence order |
| `gt_boxes` | `torch.Tensor` | `(m, 4)` | Ground-truth boxes `xyxy` |
| `gt_cls` | `torch.Tensor` | `(m,)` | Ground-truth class indices |

For each image the method computes per-class:

| Counter | Meaning |
|---|---|
| **TP** | Predictions matched to a GT box of the same class (IoU ≥ threshold) |
| **FP** | Predictions with no same-class GT match: either overlapped a GT of a different class, or had no nearby GT at all |
| **FN** | GT boxes not matched by any prediction |
| **TN** | Classes absent from both GT and predictions (correctly not detected) |

FP events are stored with the GT class at that location (`None` = no GT box nearby, displayed as `"background"`).

### `per_image_class_rows()`

Returns `list[list]` with columns `[Image, Class, TP, FP, FN, TN, "FP confused with"]`.

- Rows where TP = FP = FN = 0 are omitted (class was correctly absent).
- The `"FP confused with"` cell lists the GT class(es) at each false-positive location for that image/class pair, comma-separated. `"-"` when there were no FPs.

### `fp_detail_rows()`

Returns `list[list]` with columns `[Image, "Predicted label", "Ground-truth label"]`, one row per FP event across all images. Ground-truth label is `"background"` when the FP had no overlapping GT box.

---

## Relationship to the validator

`validate()` in `trainer/detection/engine/validator.py` creates a `DetMetrics` instance, feeds each batch through `update`, and calls `compute()` at the end. `ConfusionMatrix` is created only when `compute_confusion=True` is passed. `PerImageStats` is created only when `compute_per_image=True` is passed (enabled by default in `Model.val()`).
