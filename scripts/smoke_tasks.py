#!/usr/bin/env python3
"""Run CPU architecture smoke tests without datasets or model downloads."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.smoke import smoke_all


if __name__ == "__main__":
    print(json.dumps(smoke_all("cpu"), indent=2))
