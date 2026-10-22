# Logging — `trainer.detection.utils.logging`

## Overview

`trainer/detection/utils/logging.py` provides a module-level logger, a run-directory helper that avoids collisions, and a human-friendly time formatter.

---

## `LOGGER`

```python
from trainer.detection.utils.logging import LOGGER

LOGGER.info("training started")
LOGGER.warning("dataloader is a bottleneck")
```

A standard `logging.Logger` named `"trainer"`. Configured on first import:

- Handler: `StreamHandler` (stderr).
- Formatter: message-only (`"%(message)s"`) — no timestamp or level prefix.
- Level: `INFO`.

Handlers are added only when none already exist, so application-level logging configuration is respected if the logger was configured before importing `trainer`.

To change the log level:

```python
import logging
logging.getLogger("trainer").setLevel(logging.DEBUG)
```

---

## `increment_path(path, exist_ok=False)`

```python
from trainer.detection.utils.logging import increment_path

run_dir = increment_path("runs/exp")
# Returns Path("runs/exp") on first call,
# Path("runs/exp2") if runs/exp already exists,
# Path("runs/exp3") if both exist, etc.
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| Path` | — | Base path for the run directory |
| `exist_ok` | `bool` | `False` | If `True`, returns and reuses `path` even if it already exists |

Creates the directory (including parents) before returning. Raises `RuntimeError` if it cannot find a free name up to `path9999`.

Used by `Trainer._setup_run_dir()` to give each run a unique directory without manual naming.

---

## `format_seconds(seconds)`

```python
from trainer.detection.utils.logging import format_seconds

format_seconds(3661)   # "1h 01m"
format_seconds(125)    # "2m 05s"
format_seconds(47)     # "47s"
```

Formats an integer number of seconds as a compact human-readable string:

- ≥ 1 hour: `"<h>h <mm>m"`
- ≥ 1 minute: `"<m>m <ss>s"`
- Otherwise: `"<s>s"`

Used in the per-epoch training log line.

---

## Log output during training

The trainer prints a fixed-width table header and one row per epoch to stdout:

```
     Epoch      box      cls      dfl   mAP50  mAP50-95      P      R        LR  data%     time
     1/100    0.832    0.621    0.914   0.000     0.000   0.000  0.000  0.01000   1.2%      12s
     2/100    0.791    0.573    0.871   0.612     0.421   0.734  0.581  0.00987   1.1%      11s
    ...
```

A WARNING is emitted if the dataloader consumes more than 5 % of epoch time:

```
WARNING  dataloader is a bottleneck: 12% of epoch spent waiting for data
```

Increase `workers` or enable `cache="ram"` to address this.
