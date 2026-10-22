from pathlib import Path

import cv2
import numpy as np
import torch

from trainer.classification.data import ImageDataset, TinyClassifier, discover_training
from trainer.cli import main


def _dataset(root: Path) -> None:
    for class_name, color in (("dark", 20), ("light", 235)):
        folder = root / class_name
        folder.mkdir(parents=True)
        for index in range(4):
            image = np.full((32, 32, 3), color + index, dtype=np.uint8)
            assert cv2.imwrite(str(folder / f"{index}.png"), image)


def test_compact_classifier_dataset_is_split_per_class(tmp_path):
    _dataset(tmp_path)
    train, val, names = discover_training(tmp_path, val_fraction=0.25, seed=7)
    assert names == ["dark", "light"]
    assert len(train) == 6
    assert len(val) == 2
    assert {sample.label for sample in val} == {0, 1}


def test_classifier_forward_shape():
    model = TinyClassifier(classes=3, width=4)
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 3)


def test_classifier_uses_shared_multi_image_augmentations(tmp_path):
    _dataset(tmp_path)
    train, _, names = discover_training(tmp_path, val_fraction=0.25, seed=7)
    dataset = ImageDataset(train, 32, augment=True, num_classes=len(names), seed=3, augmentation={
        "mosaic": 1.0, "mosaic9": 0.0, "mixup": 1.0, "copy_paste": 1.0,
        "degrees": 10.0, "translate": 0.1, "scale": 0.2, "shear": 2.0,
        "perspective": 0.001, "flipud": 1.0, "fliplr": 1.0,
        "hsv_h": 0.01, "hsv_s": 0.2, "hsv_v": 0.2,
        "blur": 1.0, "noise": 1.0, "jpeg": 1.0, "erasing": 1.0,
        "square_crop": True,
    })
    image, target, _ = dataset[0]
    assert image.shape == (3, 32, 32)
    assert target.shape == (2,)
    assert torch.isclose(target.sum(), torch.tensor(1.0), atol=1e-5)
    assert (target >= 0).all()


def test_classifier_one_epoch_cpu_smoke(tmp_path):
    data = tmp_path / "data"
    output = tmp_path / "run"
    _dataset(data)
    assert main([
        "train", "task=classification", "arch=convnext-tiny", f"data={data}", f"output={output}", "epochs=1",
        "batch=2", "imgsz=32", "width=4", "device=cpu",
    ]) == 0
    assert (output / "best.pt").is_file()
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["format"] == "trainer-image-classifier-v3"
    assert checkpoint["arch"] == "convnext-tiny"
    assert checkpoint["classes"] == ["dark", "light"]
    assert (output / "weights" / "best.pt").is_file()
    assert (output / "logs" / "history.csv").is_file()
    assert (output / "reports" / "report.html").is_file()


def test_classifier_matches_detector_lifecycle_on_cpu(tmp_path):
    data = tmp_path / "data"
    run = tmp_path / "run"
    _dataset(data)
    common = ["batch=2", "imgsz=32", "width=4", "workers=0", "device=cpu"]
    assert main(["train", "task=classification", f"data={data}", f"output={run}", "epochs=1", *common]) == 0
    checkpoint = run / "weights" / "best.pt"

    evaluation = tmp_path / "evaluation"
    assert main(["evaluate", "task=classification", f"data={data}", f"checkpoint={checkpoint}",
                 f"output={evaluation}", "batch=2", "device=cpu"]) == 0
    assert (evaluation / "metrics.json").is_file()
    assert (evaluation / "per_image.csv").is_file()
    assert (evaluation / "reports" / "report.html").is_file()

    assert main(["tune-thresh", "task=classification", f"data={data}", f"checkpoint={checkpoint}",
                 f"output={evaluation}", "batch=2", "steps=8", "device=cpu"]) == 0
    thresholds = evaluation / "thresholds.json"
    assert thresholds.is_file()

    report = tmp_path / "report"
    assert main(["report", "task=classification", f"data={data}", f"checkpoint={checkpoint}",
                 f"thresholds={thresholds}", f"output={report}", "batch=2", "device=cpu"]) == 0
    assert (report / "reports" / "report.html").is_file()

    predictions = tmp_path / "predict"
    assert main(["predict", "task=classification", f"source={data}", f"checkpoint={checkpoint}",
                 f"thresholds={thresholds}", f"output={predictions}", "batch=2", "topk=2", "device=cpu"]) == 0
    assert (predictions / "predictions.csv").is_file()

    tools = tmp_path / "tools"
    assert main(["viz-augment", "task=classification", f"data={data}", f"output={tools}",
                 "imgsz=32", "n=4"]) == 0
    assert (tools / "plots" / "augment_preview.jpg").is_file()
    assert main(["bench-data", "task=classification", f"data={data}", f"output={tools}",
                 "imgsz=32", "batch=2", "batches=2"]) == 0
    assert (tools / "logs" / "benchmark.json").is_file()

    assert main(["train", "task=classification", f"data={data}", f"output={run}", "epochs=2",
                 f"resume={run / 'weights' / 'last.pt'}", *common]) == 0
    resumed = torch.load(run / "weights" / "last.pt", map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 2

    finetune = tmp_path / "finetune"
    assert main(["finetune", "task=classification", f"data={data}", f"checkpoint={checkpoint}",
                 f"output={finetune}", "epochs=1", "freeze=1", *common]) == 0
    assert (finetune / "weights" / "best.pt").is_file()

    tuning = tmp_path / "tuning"
    assert main(["tune", "task=classification", f"data={data}", f"output={tuning}",
                 "iterations=1", "tune_epochs=1", *common]) == 0
    assert (tuning / "best_hyp.json").is_file()
