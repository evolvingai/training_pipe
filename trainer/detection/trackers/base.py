"""Tracker interface. Failures must never kill a run: every call is guarded."""

from __future__ import annotations

from pathlib import Path

from trainer.detection.utils.logging import LOGGER


class Tracker:
    def log_params(self, params: dict) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...
    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None: ...
    def set_tags(self, tags: dict) -> None: ...
    def finish(self) -> None: ...
    def resume(self, run_id: str | None, start_step: int) -> None: ...

    @property
    def run_id(self) -> str | None:
        return None


class MultiTracker(Tracker):
    """Fans out to backends, swallowing (and logging) their exceptions."""

    def __init__(self, trackers: list[Tracker]):
        self.trackers = trackers

    def _each(self, method: str, *args, **kwargs) -> None:
        for t in self.trackers:
            for attempt in range(3):
                try:
                    getattr(t, method)(*args, **kwargs)
                    break
                except Exception as e:  # noqa: BLE001 - tracker failure must not kill training
                    if attempt == 2:
                        LOGGER.warning(f"tracker {type(t).__name__}.{method} failed: {e}")
                    else:
                        import time
                        time.sleep(2 ** attempt)

    def log_params(self, params: dict) -> None:
        self._each("log_params", params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._each("log_metrics", metrics, step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        self._each("log_artifact", path, artifact_path)

    def set_tags(self, tags: dict) -> None:
        self._each("set_tags", tags)

    def finish(self) -> None:
        self._each("finish")

    def resume(self, run_id: str | None, start_step: int) -> None:
        self._each("resume", run_id, start_step)

    @property
    def run_id(self) -> str | None:
        for t in self.trackers:
            if t.run_id:
                return t.run_id
        return None
