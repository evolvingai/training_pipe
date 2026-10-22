"""Logger setup and run-directory helpers."""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger("trainer")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def increment_path(path: str | Path, exist_ok: bool = False) -> Path:
    """Return path, or path2, path3, ... if it already exists."""
    path = Path(path)
    if exist_ok or not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return path
    for n in range(2, 10000):
        candidate = path.with_name(f"{path.name}{n}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError(f"could not create unique run dir for {path}")


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"
