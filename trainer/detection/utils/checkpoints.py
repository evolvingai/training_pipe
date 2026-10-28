"""Safe, versioned checkpoint loading and model reconstruction."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch


CHECKPOINT_VERSION = 4


def atomic_torch_save(value, path: str | Path) -> None:
    """Write a checkpoint atomically so an interrupted save cannot corrupt it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            tmp_path = Path(handle.name)
        torch.save(value, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def checkpoint_safe(value):
    """Convert config metadata to types accepted by ``weights_only=True``."""
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, dict):
        return {checkpoint_safe(k): checkpoint_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [checkpoint_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool, torch.Tensor)):
        return value
    raise TypeError(f"unsupported checkpoint metadata type: {type(value).__name__}")


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """Load tensor/state-only checkpoints without enabling arbitrary pickle code."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"invalid checkpoint {path}: expected a dictionary")
    version = int(checkpoint.get("checkpoint_version", 1))
    if version > CHECKPOINT_VERSION:
        raise RuntimeError(
            f"checkpoint version {version} is newer than supported version {CHECKPOINT_VERSION}"
        )
    return checkpoint


def model_from_checkpoint(checkpoint: dict, nc: int | None = None):
    """Rebuild a DetectionModel, preferring the embedded architecture."""
    from trainer.detection.nn.model import DetectionModel

    model_spec = checkpoint.get("model_cfg") or checkpoint.get("model_name")
    if model_spec is None:
        raise KeyError("checkpoint is missing both 'model_cfg' and 'model_name'")
    return DetectionModel(
        model_spec,
        nc=int(nc if nc is not None else checkpoint["nc"]),
        ch=int(checkpoint.get("ch", 3)),
        reg_max=int(checkpoint.get("reg_max", 16)),
    )
