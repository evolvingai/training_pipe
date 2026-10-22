"""Configuration helpers for the shared augmentation subsystem."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

AUGMENTATION_DEFAULTS: dict[str, Any] = {
    "aug_profile": "generic",
    "mosaic": 1.0,
    "mosaic9": 0.0,
    "mixup": 0.0,
    "mixup_alpha": 1.5,
    "copy_paste": 0.0,
    "copy_paste_iou": [0.10, 0.55],
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "brightness_contrast": 0.0,
    "brightness": 0.2,
    "contrast": 0.2,
    "blur": 0.0,
    "noise": 0.0,
    "jpeg": 0.0,
    "erasing": 0.0,
    "square_crop": False,
    "square_crop_jitter": 0.0,
    "min_box_visibility": 0.1,
}

CLASSIFICATION_AUGMENTATION_DEFAULTS: dict[str, Any] = {
    **AUGMENTATION_DEFAULTS,
    # Multi-image transforms are available but opt-in for classification.
    "mosaic": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "brightness_contrast": 1.0,
}

PASSTHROUGH_AUGMENTATION_DEFAULTS: dict[str, Any] = {
    **CLASSIFICATION_AUGMENTATION_DEFAULTS,
    "fliplr": 0.0,
    "brightness_contrast": 0.0,
}


def load_augmentation_config(path: str | Path | None) -> dict[str, Any]:
    """Load an external TOML augmentation profile.

    Values may be at the document root or under an ``[augmentation]`` table.
    """
    if not path:
        return {}
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"augmentation config does not exist: {source}")
    with source.open("rb") as handle:
        value = tomllib.load(handle)
    selected = value.get("augmentation", value)
    if not isinstance(selected, dict):
        raise ValueError(f"augmentation config must be a TOML table: {source}")
    unknown = sorted(set(selected) - set(AUGMENTATION_DEFAULTS))
    if unknown:
        raise KeyError(f"unknown augmentation option(s): {', '.join(unknown)}")
    return dict(selected)


def augmentation_config(overrides: dict[str, Any] | None = None, *,
                        defaults: dict[str, Any] | None = None,
                        config_path: str | Path | None = None) -> dict[str, Any]:
    """Merge defaults, an external profile, and explicit overrides."""
    result = dict(defaults or AUGMENTATION_DEFAULTS)
    result.update(load_augmentation_config(config_path))
    if overrides:
        result.update({key: value for key, value in overrides.items()
                       if key in AUGMENTATION_DEFAULTS and value is not None})
    return result


def add_augmentation_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the common augmentation surface on a task CLI parser."""
    parser.add_argument("--augmentation-config", default="")
    parser.add_argument("--aug-profile", choices=("generic", "topdown", "frontal"), default=None)
    for name in ("mosaic", "mosaic9", "mixup", "copy_paste", "flipud", "fliplr",
                 "brightness_contrast", "blur", "noise", "jpeg", "erasing"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=None)
    for name in ("mixup_alpha", "degrees", "translate", "scale", "shear", "perspective",
                 "hsv_h", "hsv_s", "hsv_v", "brightness", "contrast", "square_crop_jitter",
                 "min_box_visibility"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=None)
    parser.add_argument("--copy-paste-iou", type=_float_pair, default=None, metavar="LOW,HIGH")
    parser.add_argument("--square-crop", action=argparse.BooleanOptionalAction, default=None)


def augmentation_overrides(namespace: argparse.Namespace) -> dict[str, Any]:
    """Select explicitly supplied augmentation values from parsed arguments."""
    return {key: getattr(namespace, key) for key in AUGMENTATION_DEFAULTS if hasattr(namespace, key)}


def _float_pair(value: str) -> list[float]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated values")
    try:
        return [float(parts[0]), float(parts[1])]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected two comma-separated numbers") from error


__all__ = [
    "AUGMENTATION_DEFAULTS", "CLASSIFICATION_AUGMENTATION_DEFAULTS", "PASSTHROUGH_AUGMENTATION_DEFAULTS",
    "add_augmentation_arguments",
    "augmentation_config", "augmentation_overrides", "load_augmentation_config",
]
