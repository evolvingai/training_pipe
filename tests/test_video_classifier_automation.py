from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from automation.video_classifier import TinyVideoClassifier, discover_training, read_clip
from trainer.cli import main


def make_video(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 6.0, (32, 32))
    assert writer.isOpened()
    for index in range(5):
        frame = np.full((32, 32, 3), color, dtype=np.uint8)
        frame[:, index:index + 4] = 255
        writer.write(frame)
    writer.release()


def dataset(root: Path) -> Path:
    for split in ("train", "val"):
        make_video(root / split / "left" / f"{split}.avi", (20, 30, 220))
        make_video(root / split / "right" / f"{split}.avi", (220, 30, 20))
    return root


def test_video_dataset_and_temporal_model(tmp_path: Path) -> None:
    root = dataset(tmp_path / "actions")
    train, val, names = discover_training(root, 0.2, 0)
    assert names == ["left", "right"]
    assert len(train) == len(val) == 2
    clip = read_clip(train[0].path, clip_len=3, imgsz=24, random_sample=False)
    assert clip.shape == (3, 3, 24, 24)
    logits = TinyVideoClassifier(2, width=4, hidden=8)(clip.unsqueeze(0))
    assert logits.shape == (1, 2)


def test_video_training_smoke_on_cpu(tmp_path: Path) -> None:
    root = dataset(tmp_path / "actions")
    output = tmp_path / "run"
    assert main([
        "train", "task=video", "arch=r2plus1d-tiny", f"data={root}", f"output={output}", "epochs=1", "batch=2",
        "imgsz=32", "clip_len=2", "width=4", "hidden=8", "device=cpu",
    ]) == 0
    payload = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert payload["format"] == "trainer-video-classifier-v2"
    assert payload["arch"] == "r2plus1d-tiny"
    assert (output / "metrics.json").is_file()
