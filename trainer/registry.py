"""Task and architecture registries for the Trainer framework.

Built-ins are loaded lazily. External packages can expose an entry point in
``trainer.architectures`` whose callable receives this module's ``architectures``
registry, or users can pass/import modules that register through the decorator.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable, Iterable


def _key(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


@dataclass(frozen=True)
class ArchitecturePlugin:
    task: str
    name: str
    builder: Callable[..., Any]
    description: str = ""
    aliases: tuple[str, ...] = ()
    family: str = ""
    input_contract: str = ""
    output_contract: str = ""
    training_strategy: str = ""
    stages: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskPlugin:
    name: str
    actions: tuple[str, ...]
    runner: Callable[[str, dict[str, Any]], int]
    default_architecture: str
    description: str = ""


class ArchitectureRegistry:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], ArchitecturePlugin] = {}

    def register(self, plugin: ArchitecturePlugin, *, replace: bool = False) -> ArchitecturePlugin:
        task, name = _key(plugin.task), _key(plugin.name)
        canonical = ArchitecturePlugin(
            task, name, plugin.builder, plugin.description,
            tuple(_key(alias) for alias in plugin.aliases), plugin.family,
            plugin.input_contract, plugin.output_contract, plugin.training_strategy,
            tuple(plugin.stages),
        )
        keys = [(task, name), *((task, alias) for alias in canonical.aliases)]
        conflicts = [key for key in keys if key in self._items]
        if conflicts and not replace:
            raise KeyError(f"architecture already registered: {conflicts[0][0]}/{conflicts[0][1]}")
        for key in keys:
            self._items[key] = canonical
        return canonical

    def get(self, task: str, name: str) -> ArchitecturePlugin:
        ensure_builtins()
        key = (_key(task), _key(name))
        if key not in self._items:
            available = ", ".join(self.names(task)) or "none"
            raise KeyError(f"unknown architecture '{name}' for task '{task}'; available: {available}")
        return self._items[key]

    def build(self, task: str, name: str, **kwargs: Any) -> Any:
        return self.get(task, name).builder(**kwargs)

    def names(self, task: str | None = None) -> list[str]:
        ensure_builtins()
        selected = _key(task) if task else None
        return sorted({plugin.name for plugin in self._items.values() if selected is None or plugin.task == selected})

    def plugins(self) -> list[ArchitecturePlugin]:
        ensure_builtins()
        return sorted(set(self._items.values()), key=lambda plugin: (plugin.task, plugin.name))


class TaskRegistry:
    def __init__(self) -> None:
        self._items: dict[str, TaskPlugin] = {}

    def register(self, plugin: TaskPlugin, *, replace: bool = False) -> TaskPlugin:
        name = _key(plugin.name)
        if name in self._items and not replace:
            raise KeyError(f"task already registered: {name}")
        canonical = TaskPlugin(name, tuple(_key(action) for action in plugin.actions), plugin.runner,
                               _key(plugin.default_architecture), plugin.description)
        self._items[name] = canonical
        return canonical

    def get(self, name: str) -> TaskPlugin:
        ensure_builtins()
        key = _key(name)
        if key not in self._items:
            raise KeyError(f"unknown task '{name}'; available: {', '.join(self.names())}")
        return self._items[key]

    def names(self) -> list[str]:
        ensure_builtins()
        return sorted(self._items)

    def plugins(self) -> list[TaskPlugin]:
        return [self._items[name] for name in self.names()]


architectures = ArchitectureRegistry()
tasks = TaskRegistry()
_BUILTINS_LOADED = False
_EXTERNAL_LOADED = False


def register_architecture(
    task: str,
    name: str,
    *,
    description: str = "",
    aliases: Iterable[str] = (),
    family: str = "",
    input_contract: str = "",
    output_contract: str = "",
    training_strategy: str = "",
    stages: Iterable[str] = (),
):
    def decorate(builder: Callable[..., Any]) -> Callable[..., Any]:
        architectures.register(ArchitecturePlugin(
            task, name, builder, description, tuple(aliases), family,
            input_contract, output_contract, training_strategy, tuple(stages),
        ))
        return builder
    return decorate


def register_task(name: str, *, actions: Iterable[str], default_architecture: str, description: str = ""):
    def decorate(runner: Callable[[str, dict[str, Any]], int]) -> Callable[[str, dict[str, Any]], int]:
        tasks.register(TaskPlugin(name, tuple(actions), runner, default_architecture, description))
        return runner
    return decorate


def ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    importlib.import_module("trainer.architectures.detection")
    importlib.import_module("trainer.architectures.classification")
    importlib.import_module("trainer.architectures.video")
    importlib.import_module("trainer.architectures.vlm")
    importlib.import_module("trainer.tasks")


def load_external_plugins(modules: Iterable[str] = ()) -> None:
    """Load explicit modules, ``TRAINER_PLUGINS``, and installed entry points."""
    global _EXTERNAL_LOADED
    ensure_builtins()
    requested = [item.strip() for item in modules if item and item.strip()]
    requested.extend(item.strip() for item in os.getenv("TRAINER_PLUGINS", "").split(",") if item.strip())
    for module in dict.fromkeys(requested):
        importlib.import_module(module)
    if _EXTERNAL_LOADED:
        return
    _EXTERNAL_LOADED = True
    discovered = metadata.entry_points()
    entries = (discovered.select(group="trainer.architectures") if hasattr(discovered, "select")
               else discovered.get("trainer.architectures", []))
    for entry in entries:
        loaded = entry.load()
        if callable(loaded):
            loaded(architectures)


def build_architecture(task: str, name: str, **kwargs: Any) -> Any:
    return architectures.build(task, name, **kwargs)
