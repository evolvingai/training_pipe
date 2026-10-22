# Trainer — `trainer.detection.engine.trainer`

## Overview

`trainer/detection/engine/trainer.py` implements the single-GPU training loop. It handles:

- Config loading and run-directory setup.
- Model construction, optional layer freezing, and `torch.compile`.
- AMP (Automatic Mixed Precision) with gradient scaling.
- Gradient accumulation over micro-batches.
- Warmup + cosine/linear LR schedule.
- ModelEMA with warmup-aware decay ramp.
- Per-epoch validation and metric logging to all tracker backends.
- Checkpoint saving (`last.pt`, `best.pt`, optional periodic saves).
- Crash-safe resume from any `last.pt`.
- Early stopping.

---

## `Trainer`

```python
from trainer.detection.engine.trainer import Trainer

trainer = Trainer(overrides={"data": "dataset.toml", "epochs": 50})
result = trainer.train()
# result: dict from validate() — mAP50, mAP50-95, precision, recall, …
```

### Constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `overrides` | `dict \| None` | `None` | Key-value config overrides (merged on top of `default.toml`) |
| `cfg` | `dict \| None` | `None` | Pre-built config dict (skips loading when provided) |
| `model` | `DetectionModel \| None` | `None` | Pre-built model (skips `_build_model` when provided) |

### `train()` return value

Returns the `result` dict from the final `validate()` call (same as `DetMetrics.compute()` output). If no validation run was completed (e.g. early exit before the first `val_period` epoch), returns `{}`.

---

## Training loop internals

### Optimizer

Selected by `optimizer` config key:

| Value | Rule | Notes |
|---|---|---|
| `"auto"` | SGD if iterations > 10,000, otherwise AdamW | iterations = epochs × batches_per_epoch |
| `"sgd"` | SGD with Nesterov momentum | |
| `"adamw"` | AdamW | `lr0` capped at 0.002 |

Parameters are split into three groups: biases, BN/1-D params (no weight decay), and weight tensors (weight decay applied).

### Learning rate schedule

A `LambdaLR` scheduler is applied after each epoch. Two modes:

| `cos_lr` | Schedule |
|---|---|
| `true` (default) | `lrf + (1 - lrf) * 0.5 * (1 + cos(π * epoch / epochs))` |
| `false` | Linear decay from 1.0 to `lrf` |

Warmup applies only to the **first `warmup_epochs * batches_per_epoch` steps** (minimum 100 steps), linearly interpolating both LR and momentum.

### Gradient accumulation

```
accumulate = max(round(nbs / batch), 1)
```

`nbs` (nominal batch size, default 64) controls the effective batch size regardless of the actual `batch`. Gradients are accumulated over `accumulate` micro-batches before each optimizer step.

### Multiscale training

When `multiscale=true`, images are randomly resized each batch. The target size is drawn from `Uniform(lo * imgsz, hi * imgsz)` where `[lo, hi] = multiscale_range`, then rounded to the nearest multiple of the largest stride.

### Mosaic close-out

For the last `close_mosaic` epochs (default 10), `Mosaic`, `MixUp`, and `CopyPaste` augmentations are disabled on the dataloader to let the model fine-tune on clean samples.

---

## EarlyStopper (`trainer/detection/engine/trainer.py`)

```python
stopper = EarlyStopper(cfg)
should_stop = stopper.step(fitness_value, epoch)
```

Stops training when `fitness` has not improved by at least `early_stop_min_delta` for `patience` consecutive validation epochs, provided at least `early_stop_min_epochs` have elapsed.

| Config key | Default | Description |
|---|---|---|
| `patience` | 50 | Epochs without improvement before stopping |
| `early_stop_min_delta` | 0.001 | Minimum improvement to reset the counter |
| `early_stop_min_epochs` | 50 | Earliest epoch at which stopping can trigger |
| `early_stop_metric` | `"mAP50-95"` | Metric maximised (`"mAP50"`, `"mAP50-95"`, `"val_loss"`) |
| `early_stop_restore_best` | `true` | Reload `best.pt` weights into the model at the end |

---

## Checkpoints

### Saved keys in `.pt` files

| Key | Description |
|---|---|
| `epoch` | Last completed epoch index |
| `global_step` | Total optimizer steps executed |
| `model` | EMA model `state_dict` |
| `raw_model` | Non-EMA model `state_dict` |
| `optimizer` | Optimizer `state_dict` |
| `scaler` | GradScaler `state_dict` |
| `ema_meta` | EMA step count, decay_max, warmup_steps |
| `stopper` | Early stopper best/counter state |
| `best_fitness` | Best fitness value seen |
| `cfg` | Full config dict used for this run |
| `names` | `dict[int, str]` class names |
| `model_name` | String model identifier (e.g. `"trainer-n"`) |
| `nc` | Number of classes |
| `run_dir` | Path to the run directory |
| `mlflow_run_id` | MLflow run ID for resume (or `None`) |

### Checkpoint files

| File | When written |
|---|---|
| `last.pt` | Every epoch |
| `best.pt` | Every epoch where fitness improved |
| `epoch<N>.pt` | Every `save_period` epochs (if `save_period > 0`) |

---

## Resume

```bash
# Auto-resume from last.pt in the default run dir:
trainer train data=ds.toml resume=true

# Resume from an explicit checkpoint:
trainer train data=ds.toml resume=runs/exp2/last.pt
```

On resume, `_restore()` reloads model weights, optimizer state, scaler state, EMA, and stopper state. Patience counter is reset to 0 on resume (fresh slate for the current run). MLflow run is resumed via `tracker.resume(run_id, start_step)`.

---

## Config keys used by the trainer

| Key | Default | Description |
|---|---|---|
| `epochs` | 100 | Total training epochs |
| `batch` | 16 | Batch size per step |
| `autobatch` | `false` | Auto-detect max batch size for GPU memory |
| `nbs` | 64 | Nominal batch size for gradient accumulation |
| `lr0` | 0.01 | Initial learning rate |
| `lrf` | 0.01 | Final LR fraction |
| `cos_lr` | `true` | Cosine (vs. linear) LR decay |
| `momentum` | 0.937 | SGD momentum / AdamW beta1 |
| `weight_decay` | 0.0005 | Weight decay |
| `warmup_epochs` | 3.0 | Warmup duration in epochs |
| `warmup_momentum` | 0.8 | Starting momentum during warmup |
| `warmup_bias_lr` | 0.1 | Bias LR at warmup start |
| `amp` | `true` | Enable AMP (CUDA only) |
| `ema_decay` | 0.9999 | EMA max decay |
| `ema_warmup_steps` | 2000 | EMA warmup steps |
| `freeze` | 0 | Number of leading backbone layers to freeze |
| `compile` | `false` | `torch.compile` the model |
| `channels_last` | `true` | Use channels-last memory format (CUDA) |
| `val_period` | 1 | Validate every N epochs |
| `save_period` | -1 | Save periodic checkpoint every N epochs (-1 = off) |
| `multiscale` | `false` | Random multi-scale training |
| `close_mosaic` | 10 | Disable mosaic for last N epochs |
