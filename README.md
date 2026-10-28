# Training Pipe

Unified training automation for object detection, image classification,
video/action classification, and vision-language model LoRA tuning. Trainer
keeps a minimal **PyTorch + OpenCV** core for detector and classifier workflows,
with optional Hugging Face Transformers + PEFT dependencies for VLM tuning.

The public package, command, automation UI, model aliases, and deployment
services all use the `Trainer` name. Detection is one task plugin rather than
the identity of the framework.

## Workflows

| Family | Train | Evaluate | Predict | Deploy artifact |
|---|---:|---:|---:|---|
| Detection | yes | yes + report | yes | ONNX / TensorRT |
| Image classification | yes + resume/finetune/tune | yes + thresholds/report | yes + top-k/rejection/xAI | ONNX / TensorRT + class contract |
| Video/action classification | yes | yes | yes | ONNX / TensorRT + clip contract |
| VLM | PEFT LoRA | loss + perplexity + answer-token accuracy | — | ONNX / TensorRT forward graph, adapter, or merged model |

Every family now goes through the same task/architecture registry:

```bash
trainer train task=detection arch=trainer-n data=dataset.toml
trainer train task=classification arch=convnext-tiny data=/data/species
trainer train task=video arch=r2plus1d-tiny data=/data/actions clip_len=8
trainer tune task=vlm arch=hf-lora model=org/model data=/data/vlm/train.jsonl
trainer evaluate task=vlm arch=hf-lora model=org/model adapter=/runs/vlm/adapter data=/data/vlm/validation.jsonl
```

Classification has the detector lifecycle rather than a reduced side script:

```bash
trainer finetune task=classification data=/data/new-species \
  checkpoint=/runs/species/weights/best.pt freeze=4
trainer evaluate task=classification data=/data/species \
  checkpoint=/runs/species/weights/best.pt project=/runs name=species-eval
trainer tune task=classification arch=convnext-tiny data=/data/species iterations=10 tune_epochs=5

# deploy artifacts (also available as separate ONNX/TensorRT buttons in the UI)
trainer export task=classification checkpoint=/runs/species/weights/best.pt format=onnx
trainer export task=video-classification checkpoint=/runs/actions/best.pt format=tensorrt device=cuda:0
trainer export task=vlm model=org/model adapter=/runs/vlm/adapter format=onnx
```

See [`docs/CLASSIFICATION.md`](docs/CLASSIFICATION.md) for the complete command
surface and run-artifact layout.

Use `trainer list` to inspect installed plugins and `trainer smoke task=all
device=cpu` to run download-free forward/backward/optimizer checks for ConvNeXt,
R(2+1)D, and the tiny VLM-LoRA fixture. See
[`docs/ARCHITECTURE_PLUGINS.md`](docs/ARCHITECTURE_PLUGINS.md) to register an
architecture from a module or Python package entry point.

## Design philosophy

- **No framework tax.** mAP evaluation, NMS, EMA, augmentations, TOML config, HTML reporting, and CSV export are all implemented in-house. The only required runtime dependencies are PyTorch and OpenCV.
- **Architecture as data.** Model topologies are pure TOML files (`trainer/detection/cfg/models/`). Backbone, neck, and head are lists of `[from, repeats, "Module", [args]]` rows resolved through a module registry — adding a new block or topology requires no parser changes.
- **Strict configuration.** All hyperparameters live in `trainer/detection/cfg/default.toml` and are overridable as CLI `key=value` args or Python kwargs. Unknown keys raise immediately so typos never silently no-op.
- **Evaluation over training.** Validation produces COCO-protocol mAP (AP50, AP50-95), a per-image/per-class TP/FP/FN/TN breakdown, and a full false-positive detail log — enough to diagnose class confusion and coverage gaps without leaving the framework.

### Latest detector integration

The detector package is synchronized with detector commit `f4914b0` from the
`cristina/parms_fixes` line and remains exposed only through the Trainer
namespace. In addition to the standard, CSP, and FPN variants, Trainer now
registers the P2 small-object and legacy comparison architectures. The detector
also includes exact/continue/branch resume semantics, MuSGD experiments,
teacher/student feature, class-logit, and box-distribution distillation,
provenance manifests, step profiling, corrected COCO metrics/NMS, and raw-vs-EMA
evaluation selection.

```bash
# small-object architecture
trainer train task=detection model=trainer-p2-n data=dataset.toml

# MuSGD experiment
trainer train task=detection model=trainer-n data=dataset.toml \
  optimizer=musgd muon_weight=0.2 sgd_weight=1.0

# frozen teacher distillation
trainer train task=detection model=trainer-n data=dataset.toml \
  teacher=/runs/teacher/weights/best.pt \
  distill_feature_weight=1.0 distill_cls_weight=0.5
```

## Architecture

Attention-centric detector: **R-ELAN** (residual efficient layer aggregation) blocks in the backbone, an **A2Block** (area attention) bottleneck in the neck, and an anchor-free decoupled head with **DFL** (distribution focal loss) box regression. Three built-in neck variants: standard FPN, CSP-FPN, and the default ELAN-neck.

Model scales follow a depth × width multiplier scheme:

| Scale | Suffix |
|---|---|
| Nano | `-n` |
| Small | `-s` |
| Medium | `-m` |
| Large | `-l` |
| XLarge | `-x` |

---

## Install

```bash
pip install -e .              # core: torch, opencv-python
pip install -e ".[track]"     # + mlflow experiment tracking
pip install -e ".[deploy]"    # + tensorrt, onnxruntime
pip install -e ".[vlm]"       # + transformers, PEFT, accelerate, Pillow
pip install -e ".[dev]"       # + pytest, pytest-cov, ruff, mypy
```

**Requirements:** Python ≥ 3.11, PyTorch ≥ 2.1, OpenCV ≥ 4.8.

---

## Quick start

Dataset descriptor (`dataset.toml`):

```toml
path = "/data/my-dataset"      # root; splits are relative to it
train = "images/train"         # dir, txt file of paths, or list of sources
val = "images/val"

[names]
0 = "person"
1 = "cart"
```

Labels: one `.txt` per image (`labels/` mirrors `images/`), lines of `cls cx cy w h` normalized to [0, 1].

### CLI

```bash
# train (model scales: trainer-{n,s,m,l,x}; variants: trainer-csp, trainer-fpn, trainer-p2, trainer-legacy)
trainer train data=dataset.toml model=trainer-s epochs=100 batch=16 imgsz=640

# evaluate
trainer val model=runs/exp/best.pt data=dataset.toml

# inference (saves annotated images to runs/predict/)
trainer predict model=runs/exp/best.pt source=images/ conf=0.4

# finetune on a new dataset (backbone freeze + automatic head remap)
trainer finetune model=runs/exp/best.pt data=new_dataset.toml freeze=10

# export
trainer export model=runs/exp/best.pt format=onnx
trainer export model=runs/exp/best.pt format=tensorrt half=true

# hyperparameter evolution, per-class threshold tuning
trainer tune data=dataset.toml tune_iterations=30
trainer tune-thresh model=runs/exp/best.pt data=dataset.toml save=thresholds.toml

# generate evaluation report (HTML + CSV)
trainer report model=runs/exp/best.pt data=dataset.toml

# data tooling
trainer viz-augment data=dataset.toml n=16     # augmentation preview grid
trainer bench-data data=dataset.toml           # dataloader throughput
```

### Python API

```python
from trainer import Model

model = Model("runs/exp/best.pt")

# Validate — returns metrics dict including per-image TP/FP/FN/TN breakdown
result = model.val("dataset.toml")

# Predict
results = model.predict("images/", conf=0.4)
for r in results:
    print(r.path, len(r), "detections")
    img = r.plot()           # numpy BGR with boxes drawn
    txt = r.to_label_txt()   # YOLO-format label string

# Export
model.export(format="onnx", imgsz=640)
model.export(format="tensorrt", half=True)

# Per-class confidence threshold optimization
thresh = model.tune_thresholds("dataset.toml", save_path="thresholds.toml")
```

---

## Evaluation outputs

`trainer val` and `trainer report` produce:

| Output | Description |
|---|---|
| Console table | Per-class AP50 / AP50-95 at the max-F1 operating point |
| `report.html` | Standalone HTML with metadata, per-class AP, training curves, TP/FP/FN/TN tables, confusion matrix |
| `per_image_stats.csv` | Per-image, per-class TP / FP / FN / TN counts; includes the GT class at each FP location |
| `fp_detail.csv` | One row per false-positive event: image, predicted label, ground-truth label (or `background`) |

The per-image breakdown uses a **class-unconstrained** IoU match at 0.5 so cross-class confusions (predicted class A over a GT box of class B) appear as FP/FN pairs rather than being silently discarded.

---

## Configuration

All hyperparameters live in [`trainer/detection/cfg/default.toml`](trainer/detection/cfg/default.toml) and are overridable as `key=value` CLI args or Python kwargs. Key groups: training schedule, EMA, early stopping, loss weights, augmentation, data loading, inference thresholds, export, threshold tuning, and hyperparameter evolution.

Model architectures are pure TOML ([`trainer/detection/cfg/models/`](trainer/detection/cfg/models)): backbone, neck, and head are lists of `[from, repeats, "Module", [args]]` rows resolved through a module registry, so new blocks and topologies need no parser changes.

---

## Layout

```
trainer/
  registry.py   task and architecture plugin registry
  tasks.py      unified action routing
  api.py        task-neutral Python API
  cli.py        single Trainer command
  architectures/
                detection, ConvNeXt, R(2+1)D, and VLM-LoRA plugins
  detection/    detector-specific config, data, network, engine, metrics,
                export, tracking, and reporting implementation
automation/     web control plane, VM agent, and cloud bootstrap
tests/          CPU-only unit + integration suite (pytest)
scripts/        smoke_e2e.py, overfit_check.py (learning-capability gate)
docs/           reference documentation for each module
```

---

## Tests

```bash
pytest tests/ -q                      # full CPU suite
python scripts/smoke_e2e.py           # detector end-to-end pipeline smoke
python scripts/smoke_tasks.py         # ConvNeXt + R(2+1)D + VLM-LoRA CPU smokes
python scripts/overfit_check.py       # learning gate: overfit tiny synthetic set
```

---

## Documentation

See [`docs/`](docs/README.md) for full reference documentation covering the model architecture, training loop, metrics, augmentation pipeline, data loading, configuration system, and reporting.

## Training automation

[`automation/`](automation/README.md) contains a deployable web control plane and an
outbound-only VM agent for CPU or GPU training on Azure and Google Compute Engine.
It provides the same queue, VM targeting, command preview, logs, history, and
artifact flow for all four Trainer families. Video/action classification uses a
compact R(2+1)D architecture plugin. VLM tuning writes portable PEFT LoRA
adapters and can merge an adapter into its base model for deployment.
The web layer also exposes the live architecture-plugin registry, interactive
model-flow diagrams, and a sample-level xAI view. The xAI product contract is a
model result—not a filesystem browser—so manifests, metrics, paths, and generic
run artifacts are not exposed there.
Its Dataset Explorer profiles either COCO or the active family’s custom dataset
on the selected worker, then presents consistent split, class-balance, box-size,
and data-quality visualizations without copying the raw dataset to the UI. The
image classifier can use COCO instances as image-level multi-label targets, so
one scene may train several simultaneous categories while detection keeps the
original boxes.
