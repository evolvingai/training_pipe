"""TOML configuration loading, override merging, and validation."""

from __future__ import annotations

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib
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
        augmentation_path = overrides.get("augmentation_config")
        if augmentation_path:
            from trainer.augmentation.config import load_augmentation_config
            cfg = merge_overrides(cfg, load_augmentation_config(augmentation_path))
        cfg = merge_overrides(cfg, overrides)
    return cfg


def find_model_cfg(name: str | Path) -> Path:
    """Resolve a model config: explicit path, or a name in cfg/models."""
    p = Path(name)
    if p.suffix == ".toml" and p.exists():
        return p
    candidate = CFG_DIR / "models" / (p.name if p.suffix == ".toml" else f"{p.name}.toml")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"model config not found: {name}")
