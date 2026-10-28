"""Image I/O boundary: OpenCV speaks BGR, trainer speaks RGB.

Every array inside trainer is RGB — dataset samples, model input tensors, and
plotting canvases alike. These two helpers are the only places the channel
order flips, so decode and encode stay the single source of truth.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_rgb(path: str | Path) -> np.ndarray | None:
    """Read an image as HWC uint8 RGB. Returns None if unreadable, like ``cv2.imread``."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path: str | Path, image: np.ndarray) -> bool:
    """Write an RGB image to disk (OpenCV expects BGR, so flip on the way out)."""
    return cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
