#!/usr/bin/env python3
"""Compatibility entry point for the core Trainer classification workflow."""

from trainer.classification.data import *  # noqa: F403
from trainer.classification.engine import *  # noqa: F403


if __name__ == "__main__":
    raise SystemExit(main())
