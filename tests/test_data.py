import json

import numpy as np
import torch

from trainer.detection.config import load_config
from trainer.detection.data import build_dataloader, parse_dataset_toml
from trainer.detection.data.build import DistributedWeightedSampler, _resolve_split, collate_fn
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


def test_dataset_rejects_out_of_range_class(tmp_path):
    import cv2

    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    for name, cls in (("ok", 1), ("bad", 2)):
        cv2.imwrite(str(img_dir / f"{name}.jpg"), np.zeros((32, 32, 3), np.uint8))
        (lbl_dir / f"{name}.txt").write_text(f"{cls} 0.5 0.5 0.5 0.5\n")
    ds = DetectionDataset([str(img_dir / "ok.jpg"), str(img_dir / "bad.jpg")], imgsz=32, nc=2)
    assert len(ds) == 1
    assert ds.skipped_total == 1


def test_getitem_tensor_contract(tiny_dataset):
    parsed = parse_dataset_toml(tiny_dataset)
    paths, _ = _resolve_split(parsed["train"], parsed["root"])
    from trainer.augmentation import build_pipeline
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


def test_distributed_weighted_sampler_is_epoch_deterministic():
    sampler = DistributedWeightedSampler([1.0, 2.0, 3.0, 4.0], num_replicas=2, rank=1, seed=7)
    first = list(sampler)
    assert first == list(sampler) and len(first) == 2
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_optional_pycoco_comparison_uses_same_image_ids(monkeypatch, tmp_path):
    import cv2
    from trainer.detection.data import coco_sota_fixes
    from trainer.detection.engine.validator import validate
    from trainer.detection.nn.model import DetectionModel

    images = tmp_path / "images"
    images.mkdir()
    cv2.imwrite(str(images / "one.jpg"), np.zeros((32, 32, 3), dtype=np.uint8))
    annotations = tmp_path / "instances.json"
    annotations.write_text(json.dumps({
        "images": [{"id": 42, "file_name": "one.jpg", "height": 32, "width": 32}],
        "categories": [{"id": 7, "name": "thing"}],
        "annotations": [{
            "id": 1, "image_id": 42, "category_id": 7, "bbox": [8, 8, 12, 12],
            "area": 144, "iscrowd": 0,
        }],
    }))
    dataset_toml = tmp_path / "coco.toml"
    dataset_toml.write_text(
        f'path = "{tmp_path}"\n'
        'train = { format = "coco", images = "images", annotations = "instances.json" }\n'
        'val = { format = "coco", images = "images", annotations = "instances.json" }\n'
    )
    parsed = parse_dataset_toml(dataset_toml)
    cfg = load_config({
        "data": str(dataset_toml), "imgsz": 32, "batch": 1, "workers": 0,
        "compare_pycoco": True, "max_det": 5,
    })
    loader = build_dataloader(parsed, cfg, "val")
    captured = {}

    def fake_evaluate(path, predictions, max_dets, image_ids):
        captured.update(path=path, predictions=predictions, max_dets=max_dets, image_ids=image_ids)
        return {"AP": 0.2, "AP50": 0.3, "AP75": 0.1, "APs": 0.2, "APm": -1.0, "APl": -1.0}

    monkeypatch.setattr(coco_sota_fixes, "coco_evaluate", fake_evaluate)
    result = validate(DetectionModel("trainer-n", nc=1), loader, cfg, parsed["names"], device=torch.device("cpu"))
    assert captured["image_ids"] == [42]
    assert captured["max_dets"] == [1, 10, 100]
    assert all(pred["image_id"] == 42 and pred["category_id"] == 7 for pred in captured["predictions"])
    assert result["pycoco"]["AP"] == 0.2
    assert result["pycoco_delta"]["AP50"] == result["mAP50"] - 0.3
