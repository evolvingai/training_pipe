from pathlib import Path

from trainer.detection.trackers.base import MultiTracker, Tracker
from trainer.detection.trackers.csv_tracker import CSVTracker
from trainer.detection.utils.logging import LOGGER


def build_tracker(cfg: dict, run_dir: Path) -> MultiTracker:
    trackers: list[Tracker] = [CSVTracker(run_dir / "logs" / "results.csv")]
    uri = cfg.get("mlflow_tracking_uri", "")
    if cfg.get("mlflow"):
        try:
            from trainer.detection.trackers.mlflow_tracker import MLflowTracker
            trackers.append(MLflowTracker(experiment=str(cfg.get("mlflow_experiment", "trainer")),
                                          run_name=str(cfg.get("mlflow_run_name", "exp")),
                                          tracking_uri=uri))
        except Exception as e:  # noqa: BLE001 - tracker failure must not kill training
            
            LOGGER.warning(f"mlflow tracking disabled (uri={uri!r}): {e}")
    return MultiTracker(trackers)


__all__ = ["Tracker", "MultiTracker", "CSVTracker", "build_tracker"]
