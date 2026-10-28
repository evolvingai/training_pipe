"""TOML configuration loading, override merging, and validation."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

CFG_DIR = Path(__file__).resolve().parent.parent / "cfg"
DEFAULT_CFG_PATH = CFG_DIR / "default.toml"


def load_toml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as f:
        return tomllib.load(f)


def _coerce(value: Any, reference: Any, key: str) -> Any:
    """Coerce a CLI/string override to the type of the default value."""
    if reference is None or value is None:
        return value
    if isinstance(reference, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes", "on"):
                return True
            if low in ("false", "0", "no", "off"):
                return False
        raise ValueError(f"config key '{key}': cannot interpret {value!r} as bool")
    if isinstance(reference, int) and not isinstance(reference, bool):
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            return int(float(value))
    if isinstance(reference, float):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value)
    if isinstance(reference, str):
        return str(value)
    return value


def merge_overrides(base: dict[str, Any], overrides: dict[str, Any], strict: bool = True) -> dict[str, Any]:
    """Return a new dict of ``base`` updated with ``overrides``.

    Unknown keys raise in strict mode so typos never silently no-op.
    Values are coerced to the type of the default when the default is scalar.
    """
    out = dict(base)
    for key, value in overrides.items():
        if key not in base:
            if strict:
                raise KeyError(f"unknown config key: '{key}'")
            out[key] = value
            continue
        ref = base[key]
        if isinstance(ref, dict) and isinstance(value, dict):
            out[key] = merge_overrides(ref, value, strict=False)
        else:
            out[key] = _coerce(value, ref, key)
    return out


def load_config(overrides: dict[str, Any] | None = None, cfg_path: str | Path | None = None) -> dict[str, Any]:
    """Load default.toml (or ``cfg_path``) and apply overrides."""
    cfg = load_toml(cfg_path or DEFAULT_CFG_PATH)
    if overrides:
        cfg = merge_overrides(cfg, overrides)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    """Reject invalid and unsupported settings before any expensive work."""
    if cfg.get("task") != "detect":
        raise ValueError("only task='detect' is supported")
    if cfg.get("resume_mode") not in ("exact", "continue", "branch"):
        raise ValueError("resume_mode must be 'exact', 'continue' or 'branch'")
    if cfg["assignment_metric"] not in ("iou", "ciou"):
        raise ValueError("assignment_metric must be iou or ciou")
    if cfg["eval_weights"] not in ("ema", "raw"):
        raise ValueError("eval_weights must be ema or raw")
    if int(cfg["reg_channels"]) < 0:
        raise ValueError("reg_channels must be 0 (auto) or positive")
    if not (float(cfg["muon_beta"]) == -1 or 0 <= float(cfg["muon_beta"]) < 1):
        raise ValueError("muon_beta must be -1 (coupled) or in [0, 1)")
    if int(cfg["muon_ns_steps"]) < 1 or int(cfg["muon_log_interval"]) < 0:
        raise ValueError("muon_ns_steps must be positive and muon_log_interval nonnegative")
    if cfg["muon_schedule"] not in ("constant", "linear"):
        raise ValueError("muon_schedule must be constant or linear")
    if cfg["muon_schedule"] == "linear":
        if str(cfg["optimizer"]).lower() != "musgd":
            raise ValueError("Muon taper requires optimizer=musgd")
        start, end = int(cfg["muon_taper_start"]), int(cfg["muon_taper_end"])
        if not 0 <= start < end - 1 < int(cfg["epochs"]):
            raise ValueError("Muon taper requires 0 <= start < end-1 < epochs")
    gains = [float(cfg[f"distill_{term}_weight"]) for term in ("feature", "cls", "box")]
    if any(not 0 <= weight < float("inf") for weight in gains) or not 0 < float(cfg["distill_temperature"]) < float("inf"):
        raise ValueError("distillation weights must be finite/nonnegative and temperature finite/positive")
    if bool(cfg["teacher"]) != any(weight > 0 for weight in gains):
        raise ValueError("distillation requires teacher=<checkpoint> AND at least one positive distill_*_weight")
    if cfg["teacher"] and cfg["autobatch"]:
        raise ValueError("distillation requires an explicit batch (autobatch does not account for teacher memory)")
    if float(cfg["label_smoothing"]) != 0:
        raise ValueError("label_smoothing must be 0: smoothing quality targets is not supported")
    if cfg["early_stop_metric"] == "val_loss" and not cfg["val_compute_loss"]:
        raise ValueError("early_stop_metric='val_loss' requires val_compute_loss=true")
    if cfg["tune_metric"] == "val_loss" and not cfg["val_compute_loss"]:
        raise ValueError("tune_metric='val_loss' requires val_compute_loss=true")
    for key in ("epochs", "batch", "imgsz", "nbs", "tal_topk", "reg_max", "max_det"):
        if int(cfg[key]) <= 0:
            raise ValueError(f"config key '{key}' must be positive")
    if int(cfg["reg_max"]) < 2:
        raise ValueError("config key 'reg_max' must be at least 2")
    if int(cfg["workers"]) < 0:
        raise ValueError("config key 'workers' must be non-negative")
    if not 0 < float(cfg["fraction"]) <= 1:
        raise ValueError("config key 'fraction' must be in (0, 1]")
    if cfg["cache"] not in ("none", "ram", "disk"):
        raise ValueError("config key 'cache' must be one of: none, ram, disk")
    if str(cfg["optimizer"]).lower() not in ("auto", "sgd", "adamw", "musgd"):
        raise ValueError("config key 'optimizer' must be one of: auto, sgd, adamw, musgd")
    if float(cfg["muon_weight"]) < 0 or float(cfg["sgd_weight"]) < 0:
        raise ValueError("config keys 'muon_weight' and 'sgd_weight' must be non-negative")
    if cfg["early_stop_metric"] not in ("mAP50-95", "mAP50", "precision", "recall", "val_loss"):
        raise ValueError("unsupported early_stop_metric")
    for key in (
        "multiscale_prob", "label_smoothing", "mosaic", "mosaic9", "mixup", "copy_paste",
        "flipud", "fliplr", "blur", "noise", "jpeg", "erasing", "min_box_visibility",
        "conf", "val_conf", "iou", "confusion_conf", "confusion_iou",
    ):
        if not 0 <= float(cfg[key]) <= 1:
            raise ValueError(f"config key '{key}' must be in [0, 1]")
    if int(cfg["tracker_retries"]) < 0 or float(cfg["tracker_backoff"]) < 0:
        raise ValueError("tracker retry settings must be non-negative")
    bounds = cfg["multiscale_range"]
    if len(bounds) != 2 or float(bounds[0]) <= 0 or float(bounds[0]) > float(bounds[1]):
        raise ValueError("multiscale_range must be two positive ascending values")


def find_model_cfg(name: str | Path) -> Path:
    """Resolve a model config: explicit path, or a name in cfg/models."""
    p = Path(name)
    if p.suffix == ".toml" and p.exists():
        return p
    candidate = CFG_DIR / "models" / (p.name if p.suffix == ".toml" else f"{p.name}.toml")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"model config not found: {name}")
