"""Python API for the unified Trainer task and architecture flow."""

from __future__ import annotations

from typing import Any, Iterable

from trainer.registry import build_architecture, load_external_plugins, tasks
from trainer.tasks import resolve_task_name, run_task


class Trainer:
    def __init__(self, task: str = "detection", architecture: str | None = None,
                 plugins: Iterable[str] = ()):
        load_external_plugins(plugins)
        self.task = resolve_task_name(task)
        self.plugin = tasks.get(self.task)
        self.architecture = architecture or self.plugin.default_architecture

    def build(self, **kwargs: Any):
        """Construct this task's registered PyTorch architecture."""
        return build_architecture(self.task, self.architecture, **kwargs)

    def run(self, action: str, **params: Any) -> int:
        """Run a dataset action through the same dispatcher used by the CLI/UI."""
        return run_task(self.task, action, {"arch": self.architecture, **params})

    def smoke(self, device: str = "cpu") -> dict[str, Any]:
        from trainer.smoke import smoke_architecture
        return smoke_architecture(self.task, self.architecture, device)
