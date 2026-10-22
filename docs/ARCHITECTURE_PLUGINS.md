# Trainer task and architecture plugins

Trainer routes every run through one command shape:

```bash
trainer <action> task=<task> arch=<architecture> key=value ...
```

Built-in tasks and their default architectures are discoverable at runtime:

```bash
trainer list
trainer list task=classification
```

The equivalent Python API uses the same registry:

```python
from trainer import Trainer

flow = Trainer(task="classification", architecture="convnext-tiny")
model = flow.build(num_classes=12, width=32)
flow.run("train", data="/data/species", epochs=30, project="/runs", name="exp")
```

Current defaults are `trainer-n` for detection, `convnext-tiny` for image
classification, `r2plus1d-tiny` for video/action classification, and
`hf-lora` for production VLM tuning. `tiny-vlm-lora` is an offline smoke model,
not a deployment VLM.

## Register an architecture from a module

An architecture builder is an ordinary callable returning a PyTorch module.
Accept the task's required arguments and `**kwargs` so future common options do
not break the plugin.

```python
# my_company/trainer_plugins.py
import torch.nn as nn

from trainer.registry import register_architecture


@register_architecture(
    "classification",
    "company-convnet",
    description="Internal image backbone",
)
def build_company_convnet(num_classes: int, width: int = 64, **kwargs):
    return nn.Sequential(
        nn.Conv2d(3, width, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(width, num_classes),
    )
```

Load it explicitly for a run:

```bash
trainer train task=classification arch=company-convnet \
  plugin=my_company.trainer_plugins data=/data/images
```

`TRAINER_PLUGINS=my_company.trainer_plugins,another.module` loads the same
modules for every invocation.

## Register through package metadata

Installed packages can expose a callable through the
`trainer.architectures` entry-point group:

```toml
[project.entry-points."trainer.architectures"]
company = "my_company.plugin:register"
```

The callable receives the global architecture registry:

```python
from trainer.registry import ArchitecturePlugin


def register(registry):
    registry.register(ArchitecturePlugin(
        task="classification",
        name="company-convnet",
        builder=build_company_convnet,
        description="Internal image backbone",
    ))
```

## Builder contracts

| Task | Required builder inputs | Model input | Model output |
|---|---|---|---|
| `classification` | `num_classes` | `B,C,H,W` | `B,num_classes` logits |
| `video-classification` | `num_classes` | `B,T,C,H,W` | `B,num_classes` logits |
| `vlm` smoke models | `vocab_size` | images plus token IDs | token logits |

Production `hf-lora` VLM tuning is constructed by the task runner because the
base model and processor come from model metadata. It uses the same task and
architecture selection in the CLI and automation queue.

## Smoke contract

Built-ins have deterministic, download-free CPU smoke tests:

```bash
trainer smoke task=classification arch=convnext-tiny device=cpu
trainer smoke task=video arch=r2plus1d-tiny device=cpu
trainer smoke task=vlm arch=tiny-vlm-lora device=cpu
trainer smoke task=all device=cpu
```

Each smoke performs forward, backward, and optimizer steps, then verifies that
at least one trainable tensor changed. The VLM smoke additionally verifies that
only LoRA adapter parameters are trainable.
