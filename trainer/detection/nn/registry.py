"""Module registry: model TOMLs reference building blocks by registered name."""

from __future__ import annotations

MODULES: dict[str, type] = {}


def register_module(cls: type) -> type:
    MODULES[cls.__name__] = cls
    return cls


def get_module(name: str) -> type:
    if name not in MODULES:
        raise KeyError(f"unknown module '{name}'; registered: {sorted(MODULES)}")
    return MODULES[name]
