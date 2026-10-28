"""Optional MLflow tracker ([track] extra). Import only when enabled."""

from __future__ import annotations

from pathlib import Path

from trainer.detection.trackers.base import Tracker
from trainer.detection.utils.logging import LOGGER


class MLflowTracker(Tracker):
    def __init__(self, experiment: str = "trainer", run_name: str = "exp",
                 tracking_uri: str = "", nested: bool = False, run_id: str | None = None):
        import mlflow  # deferred: optional dependency

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        if run_id:
            # resume: reattach to the existing run instead of creating a new one
            self._run = mlflow.start_run(run_id=run_id)
            LOGGER.info(f"mlflow resumed run_id: {run_id}")
        else:
            self._run = mlflow.start_run(run_name=run_name, nested=nested)
            LOGGER.info(f"mlflow run_id: {self._run.info.run_id}")
            LOGGER.info(f"mlflow run name: {run_name}")
        LOGGER.info(f"mlflow experiment: {experiment}")
        LOGGER.info("MLflow initialized successfully")

    @property
    def run_id(self) -> str | None:
        return self._run.info.run_id if self._run else None

    def resume(self, run_id: str | None, start_step: int) -> None:
        if run_id and run_id != self.run_id:
            self._mlflow.end_run()
            self._run = self._mlflow.start_run(run_id=run_id)
            LOGGER.info(f"mlflow resumed run {run_id} at step {start_step}")

    def log_params(self, params: dict) -> None:
        clean = {k: str(v) for k, v in params.items() if not isinstance(v, dict)}
        # mlflow caps batch size; chunk to be safe
        items = list(clean.items())
        for i in range(0, len(items), 90):
            self._mlflow.log_params(dict(items[i:i + 90]))

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        p = Path(path)
        if p.is_dir():
            self._mlflow.log_artifacts(str(p), artifact_path=artifact_path)
        else:
            self._mlflow.log_artifact(str(p), artifact_path=artifact_path)

    def set_tags(self, tags: dict) -> None:
        self._mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def finish(self) -> None:
        self._mlflow.end_run()
