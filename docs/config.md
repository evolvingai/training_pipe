# Configuration — `trainer.detection.config`

## Overview

`trainer/detection/config/` handles TOML loading, override merging, and validation. The default configuration lives in `trainer/detection/cfg/default.toml`.

Files:
- `loader.py` — `load_toml`, `load_config`, `merge_overrides`, `find_model_cfg`.
- `writer.py` — `dump` (write a dict to TOML).

---

## Default config (`trainer/detection/cfg/default.toml`)

All keys have defaults. Any key can be overridden from the CLI or Python API. Full reference in [manual.md](manual.md#configuration-reference).

---

## `load_toml(path)` (`trainer/detection/config/loader.py`)

```python
from trainer.detection.config import load_toml

cfg = load_toml("path/to/file.toml")
# returns dict[str, Any]
```

Thin wrapper around `tomllib.load`. Raises `FileNotFoundError` / `tomllib.TOMLDecodeError` on failure.

---

## `load_config(overrides, cfg_path)` (`trainer/detection/config/loader.py`)

```python
from trainer.detection.config import load_config

cfg = load_config({"epochs": 50, "batch": 32})
# Loads default.toml, then applies overrides
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `overrides` | `dict \| None` | `None` | Keys to override in the defaults |
| `cfg_path` | `str \| Path \| None` | `None` | Use a custom base config instead of `default.toml` |

Internally calls `merge_overrides` with `strict=True`, so unknown keys raise `KeyError`.

---

## `merge_overrides(base, overrides, strict=True)` (`trainer/detection/config/loader.py`)

```python
from trainer.detection.config import merge_overrides

merged = merge_overrides(base_cfg, {"lr0": 0.005, "amp": False})
```

Returns a new dict with `base` updated by `overrides`.

**Type coercion**: override values are coerced to the type of the default:

| Default type | Coercion |
|---|---|
| `bool` | Accepts `bool` or `"true"/"false"/"1"/"0"/"yes"/"no"/"on"/"off"` |
| `int` | `int(float(value))` — accepts `"100"`, `100`, `100.0` |
| `float` | `float(value)` |
| Other | No coercion; value is used as-is |

**Strict mode** (default): unknown keys raise `KeyError`. Set `strict=False` to allow new keys (used when merging checkpoint configs at inference time).

Nested dicts are merged recursively with `strict=False`.

---

## `find_model_cfg(name)` (`trainer/detection/config/loader.py`)

```python
from trainer.detection.config import find_model_cfg

path = find_model_cfg("trainer")        # -> cfg/models/trainer.toml
path = find_model_cfg("custom.toml")    # -> explicit path if it exists
```

Resolves a model name or path to an absolute path in `cfg/models/`.

---

## `dump(cfg, path)` (`trainer/detection/config/writer.py`)

```python
from trainer.detection.config import dump

dump({"epochs": 100, "batch": 32}, "runs/exp/config.toml")
```

Serializes a config dict to TOML format. Called automatically by the trainer at the start of each run to snapshot the full resolved config.

---

## Model TOML format (`trainer/detection/cfg/models/`)

Model architecture is described in a TOML file with this structure:

```toml
nc = 80   # default number of classes

[scales]
# scale = [depth, width, max_channels]
n = [0.33, 0.25, 1024]
s = [0.33, 0.50, 1024]
m = [0.67, 0.75,  768]
l = [1.00, 1.00,  512]
x = [1.00, 1.25,  512]

[[backbone]]
# [from, repeats, module_name, [args...]]
[-1,  1, "Conv",  [64, 3, 2]]
[-1,  1, "Conv",  [128, 3, 2]]
[-1,  3, "RELAN", [128]]
...

[[head]]
[[-1, 6, 15], 1, "Detect", ["nc"]]
```

**`from`**: source layer index. `-1` = previous layer. A list = multiple inputs (for Concat/Detect).

**`repeats`**: number of times the module is stacked. Scaled by the depth multiplier.

**`module_name`**: must be registered in `trainer.detection.nn.registry`.

**`args`**: positional arguments after `(c1, c2)`. `"nc"` is replaced with the actual number of classes at build time.

---

## Dataset TOML format (`trainer/detection/cfg/dataset.example.toml`)

```toml
root = "/absolute/path/to/dataset"
nc = 3
names = {0 = "cat", 1 = "dog", 2 = "bird"}

[train]
path = "images/train"   # relative to root

[val]
path = "images/val"
```

The `root` field may be relative (resolved relative to the TOML file's directory).
