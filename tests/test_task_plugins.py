from __future__ import annotations

import json

import torch

from trainer import Trainer
from trainer.cli import main
from trainer.registry import ArchitecturePlugin, architectures, build_architecture, tasks
from trainer.smoke import smoke_all, smoke_architecture


def test_builtin_tasks_and_architectures_are_registered() -> None:
    assert {"detection", "classification", "video-classification", "vlm"} <= set(tasks.names())
    assert {"trainer-n", "trainer-csp-s", "trainer-fpn-x"} <= set(architectures.names("detection"))
    assert "convnext-tiny" in architectures.names("classification")
    assert "r2plus1d-tiny" in architectures.names("video-classification")
    assert {"tiny-vlm-lora", "hf-lora"} <= set(architectures.names("vlm"))
    assert {"finetune", "tune", "tune-thresh", "report", "viz-augment", "bench-data"} <= \
        set(tasks.get("classification").actions)


def test_architecture_registry_accepts_a_runtime_plugin() -> None:
    name = "test-linear-plugin"
    if name not in architectures.names("classification"):
        architectures.register(ArchitecturePlugin(
            "classification", name,
            lambda num_classes, **_: torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, num_classes)),
            "test plugin",
        ))
    model = build_architecture("classification", name, num_classes=2)
    assert model(torch.randn(3, 3, 2, 2)).shape == (3, 2)


def test_python_api_builds_selected_architecture() -> None:
    model = Trainer("classification", "convnext-tiny").build(num_classes=4, width=4)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 4)


def test_all_non_detector_architectures_complete_optimizer_smokes() -> None:
    results = smoke_all("cpu")
    assert [result["architecture"] for result in results] == [
        "convnext-tiny", "r2plus1d-tiny", "tiny-vlm-lora",
    ]
    assert all(result["ok"] and result["updated_tensors"] > 0 for result in results)
    assert smoke_architecture("action", "r2plus1d", "cpu")["output_shape"] == [2, 3]


def test_cli_lists_plugin_contract(capsys) -> None:
    assert main(["list", "task=classification"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["tasks"][0]["default_architecture"] == "convnext-tiny"
    convnext = next(item for item in value["architectures"] if item["name"] == "convnext-tiny")
    assert convnext["family"] == "ConvNeXt"
    assert convnext["stages"] == ["image", "stem", "hierarchical features", "global pooling", "class logits"]
    assert "Per-class logits" in convnext["output_contract"]
