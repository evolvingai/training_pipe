"""Unified Trainer CLI for detection, image/video classification, and VLMs."""

from __future__ import annotations

import json
import sys
from typing import Any

from trainer.detection.cli import parse_kv
from trainer.registry import architectures, load_external_plugins, tasks
from trainer.tasks import resolve_task_name, run_task

USAGE = """Trainer - pluggable detection, classification, action, and VLM training

usage:
  trainer <action> [task=<task>] [arch=<architecture>] [key=value ...]
  trainer list [task=<task>]
  trainer smoke [task=all] [arch=<architecture>] [device=cpu]

examples:
  trainer train task=detection arch=trainer-n data=dataset.toml epochs=100
  trainer train task=classification arch=convnext-tiny data=/data/species epochs=30
  trainer train task=video arch=r2plus1d-tiny data=/data/actions clip_len=8
  trainer tune task=vlm arch=hf-lora model=org/model data=/data/vlm/train.jsonl
  trainer smoke task=all device=cpu
"""


def _plugins(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _list_plugins(task: str | None = None) -> int:
    canonical = resolve_task_name(task) if task else None
    result = {
        "tasks": [{"name": plugin.name, "actions": plugin.actions,
                   "default_architecture": plugin.default_architecture, "description": plugin.description}
                  for plugin in tasks.plugins() if canonical is None or plugin.name == canonical],
        "architectures": [{
            "task": plugin.task,
            "name": plugin.name,
            "description": plugin.description,
            "aliases": plugin.aliases,
            "family": plugin.family,
            "input_contract": plugin.input_contract,
            "output_contract": plugin.output_contract,
            "training_strategy": plugin.training_strategy,
            "stages": plugin.stages,
        }
                          for plugin in architectures.plugins() if canonical is None or plugin.task == canonical],
    }
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] in {"help", "-h", "--help"}:
        print(USAGE)
        return 0
    action, *raw = argv
    params = parse_kv(raw)
    load_external_plugins(_plugins(params.pop("plugin", None) or params.pop("plugins", None)))
    if action == "list":
        return _list_plugins(str(params.get("task"))) if params.get("task") else _list_plugins()
    if action == "smoke":
        from trainer.smoke import smoke_all, smoke_architecture
        task = str(params.pop("task", "all"))
        device = str(params.pop("device", "cpu"))
        arch = params.pop("arch", None)
        results = smoke_all(device) if task == "all" else [smoke_architecture(task, str(arch) if arch else None, device)]
        print(json.dumps(results, indent=2))
        return 0
    task = resolve_task_name(str(params.pop("task", "detection")))
    return run_task(task, action, params)


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
