# Trackers — `trainer.detection.trackers`

## Overview

The tracker subsystem (`trainer/detection/trackers/`) provides a uniform interface for logging metrics, parameters, and artifacts during training. All tracker failures are silently caught so that a broken tracker backend never kills a training run.

Files:

- `base.py` — `Tracker` (no-op interface) and `MultiTracker` (fan-out).
- `csv_tracker.py` — `CSVTracker`: always-on local file backend.
- `mlflow_tracker.py` — `MLflowTracker`: optional MLflow backend.

---

## Tracker interface (`trainer/detection/trackers/base.py`)

```python
class Tracker:
    def log_params(self, params: dict) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...
    def log_artifact(self, path: str | Path) -> None: ...
    def set_tags(self, tags: dict) -> None: ...
    def finish(self) -> None: ...
    def resume(self, run_id: str | None, start_step: int) -> None: ...

    @property
    def run_id(self) -> str | None: ...
```

All methods are no-ops in the base class. Implement only what the backend supports.

---

## MultiTracker (`trainer/detection/trackers/base.py`)

```python
multi = MultiTracker([csv_tracker, mlflow_tracker])
multi.log_metrics({"train/box": 0.5}, step=10)
```

Fans out every call to all registered backends. Any exception raised by a backend is caught, logged at WARNING level via `LOGGER`, and does not propagate. This guarantees that experiment tracking can never interrupt training.

`run_id` property returns the first non-None `run_id` found across backends (useful for MLflow resume).

---

## CSVTracker (`trainer/detection/trackers/csv_tracker.py`)

Always active — enabled unconditionally by the trainer.

```python
from trainer.detection.trackers.csv_tracker import CSVTracker

tracker = CSVTracker("runs/exp/results.csv")
tracker.log_metrics({"train/box": 0.42, "val/mAP50": 0.75}, step=5)
```

### Behaviour

- Appends one CSV row per `log_metrics` call (one row per epoch).
- **Resume-safe**: reads existing rows on construction and skips steps already written.
- **Schema-flexible**: if a new metric key appears that is not yet in the header, the entire file is rewritten with the expanded column set and existing rows are preserved.
- Column order: `step` is always first; remaining columns are sorted alphabetically.
- `log_params`, `log_artifact`, `set_tags`, `finish`, `resume` are all no-ops — parameters live in the `config.toml` saved next to the CSV.

### Output

File path passed to constructor (e.g. `runs/exp/results.csv`):

```
step,train/box,train/cls,train/dfl,val/mAP50,val/mAP50-95,...
0,0.832145,...
1,0.751234,...
```

---

## MLflowTracker (`trainer/detection/trackers/mlflow_tracker.py`)

Optional — activated when `mlflow = true` in config. Requires the `[track]` extra (`pip install -e ".[track]"`).

```python
from trainer.detection.trackers.mlflow_tracker import MLflowTracker

tracker = MLflowTracker(experiment="my_project", run_name="run_001")
tracker.log_params({"epochs": 100, "batch": 32})
tracker.log_metrics({"val/mAP50": 0.78}, step=10)
tracker.log_artifact("runs/exp/best.pt")
tracker.finish()
```

### Behaviour

- Starts an MLflow run immediately in `__init__`; logs the run ID via `LOGGER`.
- `log_params`: flattens nested dicts, converts to strings. Chunks in groups of 90 to stay within MLflow's batch limit.
- `log_metrics`: forwards to `mlflow.log_metrics` with the same `step`.
- `log_artifact`: uploads the file to the active run's artifact store.
- `set_tags`: forwards to `mlflow.set_tags`.
- `finish`: calls `mlflow.end_run()`.
- `resume(run_id, start_step)`: if the given `run_id` differs from the current run, ends the current run and resumes the specified one.

### Configuration keys

| Key | Default | Description |
|---|---|---|
| `mlflow` | `false` | Enable MLflow tracker |
| `mlflow_experiment` | `"trainer"` | Experiment name |

---

## How trackers are built

`trainer/detection/trackers/__init__.py` exports `build_tracker(cfg, run_dir)` which:

1. Always adds `CSVTracker(run_dir / "results.csv")`.
2. If `cfg["mlflow"]` is `true`, adds `MLflowTracker(experiment=cfg["mlflow_experiment"], run_name=cfg["name"])`.
3. Returns a `MultiTracker` wrapping all collected backends.

The trainer calls `build_tracker` once at the start of training:

```python
tracker = build_tracker(cfg, self.run_dir)
tracker.log_params(cfg)
tracker.set_tags({"model": model.model_name, "dataset": str(cfg["data"])})
# ... per epoch:
tracker.log_metrics(metrics_row, step=epoch)
# at the end:
tracker.log_artifact(self.run_dir / "best.pt")
tracker.log_artifact(self.run_dir / "results.csv")
tracker.finish()
```
