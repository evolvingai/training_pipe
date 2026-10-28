"""Always-on CSV tracker: one merged row per step, resume-safe."""

from __future__ import annotations

import csv
import os
import tempfile
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
        row = {"step": step, **{k: f"{v:.6g}" for k, v in metrics.items()}}
        if self._fieldnames is None or any(k not in self._fieldnames for k in row):
            self._rewrite_with_fields(sorted(set(self._fieldnames or []) | set(row)))
        if step in self._written_steps:
            self._update_step(row)
            return
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(row)
        self._written_steps.add(step)

    def _update_step(self, update: dict) -> None:
        """Merge repeated logs for a step instead of discarding later metrics."""
        with self.path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        target = str(update["step"])
        for row in rows:
            if row.get("step") == target:
                for key, value in update.items():
                    if key not in row or row[key] == "":
                        row[key] = value
                break
        self._atomic_write(rows, self._fieldnames)

    def _rewrite_with_fields(self, fieldnames: list[str]) -> None:
        fieldnames = ["step"] + [f for f in fieldnames if f != "step"]
        old_rows: list[dict] = []
        if self.path.exists() and self._fieldnames:
            with self.path.open() as f:
                old_rows = list(csv.DictReader(f))
        self._atomic_write(old_rows, fieldnames)
        self._fieldnames = fieldnames

    def _atomic_write(self, rows: list[dict], fieldnames: list[str] | None) -> None:
        """Replace the CSV only after a complete sibling file has been flushed."""
        assert fieldnames is not None
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", newline="", encoding="utf-8", prefix=f".{self.path.name}.",
                suffix=".tmp", dir=self.path.parent, delete=False,
            ) as handle:
                tmp_path = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

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
