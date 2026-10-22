import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory) -> Path:
    """Deterministic 2-class synthetic dataset; returns the dataset TOML path."""
    root = tmp_path_factory.mktemp("dataset")
    rng = np.random.default_rng(7)
    size = 96
    for split, n in (("train", 8), ("val", 4)):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(n):
            img = np.full((size, size, 3), 25, dtype=np.uint8)
            lines = []
            for _ in range(int(rng.integers(1, 3))):
                cls = int(rng.integers(0, 2))
                w, h = int(rng.integers(20, 40)), int(rng.integers(20, 40))
                x = int(rng.integers(0, size - w))
                y = int(rng.integers(0, size - h))
                color = (0, 0, 220) if cls == 0 else (0, 220, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
                lines.append(f"{cls} {(x + w / 2) / size:.6f} {(y + h / 2) / size:.6f} "
                             f"{w / size:.6f} {h / size:.6f}")
            cv2.imwrite(str(img_dir / f"{i:03d}.jpg"), img)
            (lbl_dir / f"{i:03d}.txt").write_text("\n".join(lines))
    toml_path = root / "dataset.toml"
    toml_path.write_text(
        f'path = "{root}"\ntrain = "images/train"\nval = "images/val"\n\n'
        '[names]\n0 = "red"\n1 = "green"\n'
    )
    return toml_path


@pytest.fixture()
def clean_runs(tmp_path) -> Path:
    d = tmp_path / "runs"
    yield d
    shutil.rmtree(d, ignore_errors=True)
