# Trainer Documentation

## Index

| Document | Description |
|---|---|
| [ARCHITECTURE_PLUGINS.md](ARCHITECTURE_PLUGINS.md) | Unified task registry, architecture-builder contract, external module/entry-point plugins, and CPU smoke commands |
| [CLASSIFICATION.md](CLASSIFICATION.md) | Image-classification train, resume, finetune, evaluate, threshold tuning, reports, prediction, tuning, data tools, and export |
| [manual.md](manual.md) | User manual — installation, dataset format, CLI commands, Python API, run directory layout, config reference |
| [model.md](model.md) | Neural network architecture — `DetectionModel`, `parse_model`, building blocks (RELAN, CSP, Detect, DFL), module registry |
| [trainer.md](trainer.md) | Training loop — AMP, EMA, LR schedule, gradient accumulation, early stopping, checkpoints, resume |
| [engine.md](engine.md) | Operational engines — validator, predictor (`Results`), ONNX/TensorRT exporter, threshold tuner |
| [trackers.md](trackers.md) | Experiment tracking — `Tracker` interface, `MultiTracker`, `CSVTracker`, `MLflowTracker` |
| [reporting.md](reporting.md) | Report generation — `report.html`, `per_image_stats.csv`, `fp_detail.csv`; per-class AP, training curves, per-image TP/FP/FN/TN, FP detail, confusion matrix |
| [metrics.md](metrics.md) | Detection metrics — `DetMetrics`, `ConfusionMatrix`, `PerImageStats`, `match_predictions`, COCO-protocol AP |
| [augmentation.md](augmentation.md) | Augmentation pipeline — geometric transforms (Mosaic, MixUp, CopyPaste, perspective), photometric transforms |
| [data.md](data.md) | Data loading — `DetectionDataset`, label cache, image cache modes, dataloader construction |
| [config.md](config.md) | Configuration system — `load_config`, `merge_overrides`, model TOML format, dataset TOML format |
| [utils.md](utils.md) | Utilities — box conversions, IoU, NMS, `ModelEMA`, plotting, device checks |
