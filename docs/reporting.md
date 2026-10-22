# Reporting — `trainer.detection.reporting`

## Overview

`trainer/detection/reporting/report.py` generates a self-contained HTML evaluation report that can be dropped into Confluence or opened directly in a browser. All plots are embedded as base64-encoded PNG so the file has no external dependencies.

---

## `generate_report` (`trainer/detection/reporting/report.py`)

```python
from trainer.detection.reporting import generate_report

out_path = generate_report(
    run_dir="runs/exp",
    result=val_result,          # dict returned by validate()
    names={0: "cat", 1: "dog"},
    cfg={"model": "trainer-n", "data": "ds.toml", "imgsz": 640},
    thresholds=thresh_result,   # optional: output of tune_thresholds()
    out_name="report.html",
)
# out_path == Path("runs/exp/report.html")
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `run_dir` | `str \| Path` | Directory containing `results.csv` and where `report.html` will be written |
| `result` | `dict` | Validation result dict (output of `validate()`) |
| `names` | `dict[int, str]` | Class index → name mapping |
| `cfg` | `dict` | Config snapshot (used for metadata section) |
| `thresholds` | `dict \| None` | Output of `tune_thresholds()`. If provided, adds a per-class threshold table |
| `out_name` | `str` | Output filename inside `run_dir` (default: `"report.html"`) |

### Report sections

1. **Metadata header** — generated timestamp, model name, dataset, image size, summary mAP50, mAP50-95, precision, recall.

2. **Per-class AP table** — one row per class that appeared in the validation set. Classes with AP50 < 0.5 are highlighted in yellow and a note suggests adding data or reviewing thresholds. Includes a final "all" row showing the mean.

3. **Per-class confidence thresholds** (optional) — included when `thresholds` is provided. Shows per-class threshold, best fitness score, precision, recall, and status (`ok` or `poorly_separated`). Rows where status ≠ `ok` are highlighted.

4. **Training curves** — reads `results.csv` from `run_dir`. Plots:
   - Loss curves: `train/box`, `train/cls`, `train/dfl` vs. step.
   - mAP curves: `val/mAP50`, `val/mAP50-95` vs. step.
   Each plot is embedded as a base64 PNG via `trainer.detection.utils.plotting.plot_curve`.

5. **Per-image per-class TP / FP / FN / TN** — included when `result["per_image_stats"]` is present (set `compute_per_image=True` in `validate()`, enabled by default in `Model.val()`). One row per `(image, class)` pair that had any detection activity; rows where the class was correctly absent are omitted. Rows with at least one FP are highlighted. The "FP confused with" column lists the GT class(es) at each false-positive location, or `"background"` when the prediction had no overlapping GT box.

6. **False-positive detail** — included alongside section 5 when any FPs exist. One row per FP event across all images, showing the predicted label and the ground-truth label at that location.

7. **Confusion matrix** — included when `result["confusion"]` is present (set `compute_confusion=True` in `validate()`). Rows = predicted class, columns = actual class, last index = background (missed / false positive).

### Output files

| File | Always written | Condition |
|---|---|---|
| `report.html` | yes | — |
| `per_image_stats.csv` | when `per_image_stats` present | columns: Image, Class, TP, FP, FN, TN, FP confused with |
| `fp_detail.csv` | when FPs exist | columns: Image, Predicted label, Ground-truth label |

All files are written inside `run_dir`. `LOGGER.info` prints the path of each file on completion.

---

## CLI usage

```bash
# Generate report with default settings
trainer report model=runs/exp/best.pt data=dataset.toml

# Include per-class thresholds in the report
trainer report model=runs/exp/best.pt data=dataset.toml thresholds=thresholds.toml

# Custom output directory
trainer report model=runs/exp/best.pt data=dataset.toml run_dir=my_report_dir
```

---

## Python API usage

```python
from trainer.detection.api import Model
from trainer.detection.data import parse_dataset_toml
from trainer.detection.reporting import generate_report
from trainer.detection.config import load_config

model = Model("runs/exp/best.pt")
# compute_per_image=True is the default in Model.val(); per_image_stats is included in result
result = model.val("dataset.toml")
parsed = parse_dataset_toml("dataset.toml")
cfg = load_config({"data": "dataset.toml"})

out_dir = Path("runs/exp")
out_dir.mkdir(parents=True, exist_ok=True)
generate_report(
    run_dir=out_dir,
    result=result,
    names=parsed["names"],
    cfg=cfg,
)
# Writes: runs/exp/report.html
#         runs/exp/per_image_stats.csv
#         runs/exp/fp_detail.csv  (if any FPs found)
```

---

## `result` dict keys used by the report

| Key | Type | Source | Optional |
|---|---|---|---|
| `mAP50` | `float` | `DetMetrics.compute()` | no |
| `mAP50-95` | `float` | `DetMetrics.compute()` | no |
| `precision` | `float` | `DetMetrics.compute()` | no |
| `recall` | `float` | `DetMetrics.compute()` | no |
| `ap50_per_class` | `np.ndarray (nc,)` | `DetMetrics.compute()` | no |
| `ap_per_class` | `np.ndarray (nc,)` | `DetMetrics.compute()` | no |
| `classes_seen` | `np.ndarray (nc,) bool` | `DetMetrics.compute()` | no |
| `confusion` | `np.ndarray (nc+1, nc+1)` | `ConfusionMatrix.matrix` | yes — `compute_confusion=True` |
| `per_image_stats` | `PerImageStats` | `PerImageStats` accumulator | yes — `compute_per_image=True` (default in `Model.val()`) |
