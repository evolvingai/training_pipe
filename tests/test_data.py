import numpy as np
import torch

from trainer.detection.config import load_config
from trainer.detection.data import build_dataloader, parse_dataset_toml
from trainer.detection.data.build import _resolve_split, collate_fn
from trainer.detection.data.dataset import DetectionDataset, img2label_path


def test_img2label_path():
    assert img2label_path("/data/images/train/a.jpg") == "/data/labels/train/a.txt"
    assert img2label_path("/data/images/a.png") == "/data/labels/a.txt"


def test_parse_dataset_toml(tiny_dataset):
    parsed = parse_dataset_toml(tiny_dataset)
    assert parsed["nc"] == 2
    assert parsed["names"] == {0: "red", 1: "green"}
    paths, _ = _resolve_split(parsed["train"], parsed["root"])
    assert len(paths) == 8


def test_dataset_loads_and_caches(tiny_dataset):
    parsed = parse_dataset_toml(tiny_dataset)
    paths, _ = _resolve_split(parsed["train"], parsed["root"])
    ds1 = DetectionDataset(paths, imgsz=96)
    assert len(ds1) == 8
    cache_files = list((parsed["root"] / "images" / "train").glob(".trainer_cache_*.npz"))
    assert len(cache_files) == 1
    # second construction hits the cache and yields identical labels
    ds2 = DetectionDataset(paths, imgsz=96)
    assert len(ds2) == len(ds1)
    for a, b in zip(ds1.labels, ds2.labels):
        assert np.allclose(a, b)


def test_dataset_skips_corrupt_image(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    import cv2
    ok = img_dir / "ok.jpg"
    cv2.imwrite(str(ok), np.zeros((32, 32, 3), np.uint8))
    bad = img_dir / "bad.jpg"
    bad.write_bytes(b"not an image")
    ds = DetectionDataset([str(ok), str(bad)], imgsz=32)
    assert len(ds) == 1
    assert ds.skipped_total == 1


def test_getitem_tensor_contract(tiny_dataset):
    parsed = parse_dataset_toml(tiny_dataset)
    paths, _ = _resolve_split(parsed["train"], parsed["root"])
    from trainer.detection.augment import build_pipeline
    cfg = load_config({"imgsz": 96, "mosaic": 0.0})
    ds = DetectionDataset(paths, imgsz=96, transforms=build_pipeline(cfg, train=True, seed=0))
    item = ds[0]
    assert item["img"].dtype == torch.uint8
    assert item["img"].shape == (3, 96, 96)
    assert item["bboxes"].shape[1] == 4
    if item["bboxes"].numel():
        assert item["bboxes"].min() >= 0 and item["bboxes"].max() <= 1


def test_collate(tiny_dataset):
    parsed = parse_dataset_toml(tiny_dataset)
    cfg = load_config({"imgsz": 96, "batch": 4, "workers": 0, "mosaic": 0.0})
    loader = build_dataloader(parsed, cfg, "train")
    batch = next(iter(loader))
    assert batch["img"].shape == (4, 3, 96, 96)
    assert batch["batch_idx"].shape[0] == batch["cls"].shape[0] == batch["bboxes"].shape[0]
    assert batch["batch_idx"].max() < 4


def test_collate_empty_labels():
    items = [{"img": torch.zeros(3, 8, 8, dtype=torch.uint8), "cls": torch.zeros(0),
              "bboxes": torch.zeros(0, 4), "path": "x", "orig_shape": (8, 8),
              "ratio": 1.0, "pad": (0.0, 0.0)}]
    batch = collate_fn(items)
    assert batch["cls"].shape[0] == 0


def test_balanced_sampling_builds(tiny_dataset):
    parsed = parse_dataset_toml(tiny_dataset)
    cfg = load_config({"imgsz": 96, "batch": 4, "workers": 0, "balanced_sampling": True,
                       "mosaic": 0.0})
    loader = build_dataloader(parsed, cfg, "train")
    batch = next(iter(loader))
    assert batch["img"].shape[0] == 4
