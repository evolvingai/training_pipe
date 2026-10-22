# Image classification

Image classification uses the same Trainer lifecycle as detection and accepts
any architecture registered for the `classification` task. The built-in
default is `convnext-tiny`.

## Dataset

Use explicit class-folder splits:

```text
dataset/
  train/<class>/*.{jpg,png,...}
  val/<class>/*.{jpg,png,...}
```

Or place `<class>` folders directly under `dataset/`; Trainer creates a
deterministic per-class validation split.

## Lifecycle commands

```bash
# Train with EMA, AMP on CUDA, early stopping, and structured checkpoints
trainer train task=classification arch=convnext-tiny data=/data/species \
  project=/runs name=species epochs=30 batch=32 imgsz=224

# Use the shared external augmentation pipeline (also used by detection/video/VLM)
trainer train task=classification arch=convnext-tiny data=/data/species \
  augmentation_config=/data/augmentations/robust.toml \
  mosaic=0.25 mixup=0.2 copy_paste=0.1 blur=0.1

# Resume optimizer, scheduler, scaler, EMA, and history
trainer train task=classification data=/data/species project=/runs name=species \
  epochs=60 resume=/runs/species/weights/last.pt

# Transfer weights to a new class set and freeze the first four feature stages
trainer finetune task=classification data=/data/new-species \
  checkpoint=/runs/species/weights/best.pt project=/runs name=species-ft freeze=4

# Validation metrics, per-image output, confusion matrix, and HTML report
trainer evaluate task=classification data=/data/species \
  checkpoint=/runs/species/weights/best.pt project=/runs name=species-eval

# Optimize one-vs-rest confidence thresholds per class
trainer tune-thresh task=classification data=/data/species \
  checkpoint=/runs/species/weights/best.pt project=/runs name=species-eval

# Rebuild a report and include the tuned thresholds
trainer report task=classification data=/data/species \
  checkpoint=/runs/species/weights/best.pt \
  thresholds=/runs/species-eval/thresholds.json project=/runs name=species-eval

# Predict top-k classes and reject results below their configured threshold
trainer predict task=classification source=/data/incoming \
  checkpoint=/runs/species/weights/best.pt \
  thresholds=/runs/species-eval/thresholds.json project=/runs name=species-predict

# Hyperparameter trials, augmentation visualization, and data throughput
trainer tune task=classification arch=convnext-tiny data=/data/species \
  project=/runs name=species-tune iterations=10 tune_epochs=5
trainer viz-augment task=classification data=/data/species project=/runs name=species-tools
trainer bench-data task=classification data=/data/species project=/runs name=species-tools

# Dynamic-batch ONNX plus JSON preprocessing/class contract
trainer export task=classification checkpoint=/runs/species/weights/best.pt \
  project=/runs name=species-onnx
```

## Run artifacts

```text
run/
  cfg/args.json
  cfg/augmentation.json
  weights/best.pt
  weights/last.pt
  weights/epoch_<n>.pt       # when save_period is enabled
  logs/results.csv           # tracker-compatible metrics
  logs/history.csv
  plots/confusion_matrix.png
  plots/loss_curve.png
  plots/accuracy_curve.png
  reports/report.html
  classes.json
  metrics.json
```

Checkpoints include raw and EMA model weights, optimizer, scheduler, AMP scaler,
EMA state, epoch history, architecture parameters, class names, and validation
metrics. Finetuning transfers every shape-compatible tensor and safely skips a
classification head whose class count changed.

Classification consumes `trainer.augmentation` directly. Geometric and sensor
transforms are shared with detection; Mosaic, MixUp, and box-free CopyPaste
(CutMix semantics) produce probability targets and use soft-target cross entropy.

Evaluation reports accuracy, macro precision/recall/F1, per-class
precision/recall/F1/support, TP/FP/FN/TN counts, the complete confusion matrix,
and per-image predictions.
