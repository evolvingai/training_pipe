"""OpenCV-based visualization: labeled batches, predictions, simple curves."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from trainer.detection.utils.image import imwrite_rgb

_PALETTE = [  # RGB, matching the rest of the pipeline
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
    (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
    (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
    (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def color_for(idx: int) -> tuple[int, int, int]:
    return _PALETTE[int(idx) % len(_PALETTE)]


def draw_boxes(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray,
               names: dict[int, str] | None = None, confs: np.ndarray | None = None) -> np.ndarray:
    """Draw xyxy boxes with class labels on an RGB image (copy)."""
    img = image.copy()
    for i, (box, cls) in enumerate(zip(boxes, classes)):
        c = int(cls)
        color = color_for(c)
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = names.get(c, str(c)) if names else str(c)
        if confs is not None:
            label = f"{label} {confs[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img, label, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def make_grid(images: list[np.ndarray], cols: int = 4) -> np.ndarray:
    """Tile equally-sized RGB images into a grid."""
    if not images:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    h, w = images[0].shape[:2]
    cols = min(cols, len(images))
    rows = (len(images) + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    return grid


def _draw_rotated_label(canvas: np.ndarray, text: str, cx: int, cy: int,
                        font: int, scale: float, color: tuple, thickness: int) -> None:
    """Draw text rotated 90° counter-clockwise (for y-axis labels)."""
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    tmp = np.full((th + 8, tw + 8, 3), 255, dtype=np.uint8)
    cv2.putText(tmp, text, (4, th + 3), font, scale, color, thickness, cv2.LINE_AA)
    rotated = cv2.rotate(tmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rh, rw = rotated.shape[:2]
    y0, x0 = cy - rh // 2, cx - rw // 2
    y1, x1 = y0 + rh, x0 + rw
    cy0, cy1 = max(y0, 0), min(y1, canvas.shape[0])
    cx0, cx1 = max(x0, 0), min(x1, canvas.shape[1])
    if cy1 > cy0 and cx1 > cx0:
        canvas[cy0:cy1, cx0:cx1] = rotated[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]


def plot_curve(x: np.ndarray, ys: dict[str, np.ndarray], out_path: str | Path,
               title: str = "", xlabel: str = "", ylabel: str = "",
               size: tuple[int, int] = (900, 600)) -> Path:
    """Render simple line curves to a PNG with OpenCV (no matplotlib)."""
    w, h = size
    ml, mr, mt, mb = 80, 40, 50, 60  # left/right/top/bottom margins
    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    all_y = np.concatenate([np.asarray(y, dtype=np.float64) for y in ys.values()]) if ys else np.array([0.0])
    y_min, y_max = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    if y_max - y_min < 1e-9:
        y_max = y_min + 1.0
    x = np.asarray(x, dtype=np.float64)
    x_min, x_max = (float(x.min()), float(x.max())) if x.size else (0.0, 1.0)
    if x_max - x_min < 1e-9:
        x_max = x_min + 1.0

    plot_w = w - ml - mr
    plot_h = h - mt - mb

    def to_px(xv: float, yv: float) -> tuple[int, int]:
        px = ml + (xv - x_min) / (x_max - x_min) * plot_w
        py = mt + (1.0 - (yv - y_min) / (y_max - y_min)) * plot_h
        return int(px), int(py)

    # grid lines + y-axis tick labels
    cv2.rectangle(canvas, (ml, mt), (w - mr, h - mb), (180, 180, 180), 1)
    for gi in range(5):
        yv = y_min + gi / 4 * (y_max - y_min)
        _, py = to_px(x_min, yv)
        cv2.line(canvas, (ml, py), (w - mr, py), (235, 235, 235), 1)
        tick_lbl = f"{yv:.3g}"
        (tw, _), _ = cv2.getTextSize(tick_lbl, font, 0.4, 1)
        cv2.putText(canvas, tick_lbl, (ml - tw - 6, py + 4), font, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

    # x-axis tick labels
    for gi in range(5):
        xv = x_min + gi / 4 * (x_max - x_min)
        px, _ = to_px(xv, y_min)
        tick_lbl = f"{xv:.3g}"
        (tw, _), _ = cv2.getTextSize(tick_lbl, font, 0.4, 1)
        cv2.putText(canvas, tick_lbl, (px - tw // 2, h - mb + 16), font, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

    # data series
    for ci, (label, y) in enumerate(ys.items()):
        y = np.asarray(y, dtype=np.float64)
        color = color_for(ci)
        pts = [to_px(xv, yv) for xv, yv in zip(x, y) if np.isfinite(yv)]
        for p1, p2 in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (ml + 10, mt + 20 + 18 * ci), font, 0.5, color, 1, cv2.LINE_AA)

    # axis labels
    if xlabel:
        (tw, _), _ = cv2.getTextSize(xlabel, font, 0.55, 1)
        cv2.putText(canvas, xlabel, ((w - tw) // 2, h - 8), font, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    if ylabel:
        _draw_rotated_label(canvas, ylabel, 12, mt + plot_h // 2, font, 0.55, (60, 60, 60), 1)

    if title:
        cv2.putText(canvas, title, (ml, 32), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite_rgb(out_path, canvas)
    return out_path
