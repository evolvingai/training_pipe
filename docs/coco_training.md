# Training on COCO 2017

Step-by-step guide for training a detector on COCO 2017 with the SOTA fixes applied.

---

## 1. Download COCO 2017

```bash
mkdir -p /data/coco
cd /data/coco

# Images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip && unzip val2017.zip

# Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

Expected layout:

```
/data/coco/
├── images/
│   ├── train2017/   (118 287 images)
│   └── val2017/     (5 000 images)
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

---

## 2. Create a dataset descriptor

Save as `coco.toml` (anywhere — paths resolve relative to it or to `path`):

```toml
path = "/data/coco"

[train]
format      = "coco"
images      = "images/train2017"
annotations = "annotations/instances_train2017.json"

[val]
format      = "coco"
images      = "images/val2017"
annotations = "annotations/instances_val2017.json"
```

`names` is optional — it is derived from the JSON `categories` array automatically.

---

## 3. Train

```bash
trainer train data=coco.toml model=trainer-n epochs=300 batch=16 imgsz=640
```

Common overrides:

| Key | Recommended for COCO | Description |
|-----|----------------------|-------------|
| `model` | `trainer-n` / `trainer-s` | Architecture size |
| `epochs` | `300` | COCO converges slowly |
| `batch` | `16` (1 GPU) | Increase with multi-GPU or larger VRAM |
| `imgsz` | `640` | Standard COCO input size |
| `workers` | `8` | DataLoader worker processes |
| `cache` | `"ram"` or `"disk"` | Speeds up I/O-bound runs |
| `copy_paste` | `0.3` | Copy-paste augmentation (SOTA fix #12) |
| `cls_loss` | `"focal"` | Focal loss for noisy labels (SOTA fix #7) |
| `label_smoothing` | `0.05` | Label smoothing for ambiguous classes (SOTA fix #10) |
| `balanced_sampling` | `true` | Inverse-frequency sampling (SOTA fix #8) |

### Python API

```python
from trainer.detection.api import Model

model = Model("trainer-n")
model.train(
    "coco.toml",
    epochs=300,
    batch=16,
    imgsz=640,
    copy_paste=0.3,
    cls_loss="focal",
    label_smoothing=0.05,
)
```

---

## 4. Validate (in-house mAP)

```bash
trainer val model=runs/exp/best.pt data=coco.toml
```

Reports mAP50 and mAP50-95 using the in-house evaluator. Per-class AP is printed to the console.

---

## 5. Evaluate with pycocotools (AP_S / AP_M / AP_L)

The in-house validator does not report AP_S / AP_M / AP_L. Use `coco_sota_fixes.coco_evaluate` for the full COCO metric suite after converting predictions to COCO format:

```python
from trainer.detection.api import Model
from trainer.detection.coco_sota_fixes import coco_evaluate

model = Model("runs/exp/best.pt")
results = model.predict("images/val2017/", conf=0.001, max_det=300)

predictions = []
for r in results:
    image_id = int(Path(r.path).stem)  # COCO filenames are zero-padded IDs
    for box, score, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
        x1, y1, x2, y2 = box.tolist()
        predictions.append({
            "image_id":   image_id,
            "category_id": int(cls),   # must match COCO category IDs
            "bbox":        [x1, y1, x2 - x1, y2 - y1],
            "score":       float(score),
        })

metrics = coco_evaluate("annotations/instances_val2017.json", predictions)
print(metrics)
# {"AP": 0.42, "AP50": 0.61, "AP75": 0.46, "APs": 0.24, "APm": 0.46, "APl": 0.57}
```

`coco_evaluate` sets `maxDets=[1, 10, 300]` (SOTA fix #4), correctly handles `iscrowd` and `ignore` regions (SOTA fix #5/#6), and returns AP_S / AP_M / AP_L (SOTA fix #11).

---

## 6. Optional: repeat factor sampler for long-tail balance

`balanced_sampling = true` in config uses inverse-frequency weighting. For the LVIS-style repeat factor sampler (stronger for rare classes) wire it in manually:

```python
from trainer.detection.coco_sota_fixes import repeat_factors, RepeatFactorSampler
from trainer.detection.data.coco_dataset import parse_coco_json
from torch.utils.data import DataLoader

_, _, category_map = parse_coco_json("annotations/instances_train2017.json")

# Build category-IDs-per-image list
import json
data = json.loads(open("annotations/instances_train2017.json").read())
img_to_cats = {}
for ann in data["annotations"]:
    if ann.get("iscrowd", 0):
        continue
    img_to_cats.setdefault(ann["image_id"], set()).add(ann["category_id"])
cat_ids_per_image = [list(v) for v in img_to_cats.values()]

factors = repeat_factors(cat_ids_per_image, t=0.001)
sampler = RepeatFactorSampler(factors, seed=0)
# loader = DataLoader(dataset, sampler=sampler, batch_size=16)
```

---

## 7. SOTA fixes applied automatically

The following fixes are active with no extra configuration:

| Fix | What happens automatically |
|-----|---------------------------|
| **#6 iscrowd filter** | `parse_coco_json` skips `iscrowd=1` annotations before they reach the label pipeline |
| **#13 Letterbox resizing** | All images are letterbox-padded (not stretched) in the augmentation pipeline |
| **#12 Copy-paste** | Available via `copy_paste=0.3` in config (default 0.0, i.e. off) |
| **#7 Focal loss** | Available via `cls_loss="focal"` in config (default `"bce"`) |
| **#10 Label smoothing** | Available via `label_smoothing=0.05` in config (default `0.0`) |
| **#4 maxDets=300** | NMS is capped at `max_det=300` by default |

---

## 8. Generate an evaluation report

```bash
trainer report model=runs/exp/best.pt data=coco.toml run_dir=runs/coco_report
```

Produces an HTML report with per-class AP tables, precision-recall curves, and confusion matrix.
