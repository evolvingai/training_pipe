"""Always-on CSV tracker: one row per step, append mode, resume-safe."""

from __future__ import annotations

import csv
from pathlib import Path

from trainer.detection.trackers.base import Tracker


class CSVTracker(Tracker):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None
        self._written_steps: set[int] = set()
        if self.path.exists():
            with self.path.open() as f:
                reader = csv.DictReader(f)
                self._fieldnames = list(reader.fieldnames or [])
                for row in reader:
                    if row.get("step"):
                        self._written_steps.add(int(float(row["step"])))

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if step in self._written_steps:
            return  # resume: skip already-written steps
        row = {"step": step, **{k: f"{v:.6g}" for k, v in metrics.items()}}
        if self._fieldnames is None or any(k not in self._fieldnames for k in row):
            self._rewrite_with_fields(sorted(set(self._fieldnames or []) | set(row)))
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(row)
        self._written_steps.add(step)

    def _rewrite_with_fields(self, fieldnames: list[str]) -> None:
        fieldnames = ["step"] + [f for f in fieldnames if f != "step"]
        old_rows: list[dict] = []
        if self.path.exists() and self._fieldnames:
            with self.path.open() as f:
                old_rows = list(csv.DictReader(f))
        with self.path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in old_rows:
                writer.writerow(row)
        self._fieldnames = fieldnames

    def log_params(self, params: dict) -> None:
        pass  # params live in the saved config TOML next to the CSV

    def log_artifact(self, path, artifact_path=None) -> None:
        pass

    def set_tags(self, tags: dict) -> None:
        pass

    def finish(self) -> None:
        pass

    def resume(self, run_id, start_step) -> None:
        pass
