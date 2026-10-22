"""Integration smoke: 2-epoch CPU training, resume, val, predict, thresholds, report."""

import torch

from trainer.detection.api import Model

COMMON = dict(imgsz=96, batch=4, workers=0, device="cpu", amp=False, mosaic=0.0,
              warmup_epochs=0.5, patience=100, deterministic=True)


def test_train_val_predict_cycle(tiny_dataset, clean_runs):
    model = Model("trainer-n")
    result = model.train(str(tiny_dataset), epochs=2, project=str(clean_runs), name="t", **COMMON)
    assert "mAP50" in result
    run_dir = clean_runs / "t"
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "last.pt").exists()
    assert (run_dir / "config.toml").exists()
    assert (run_dir / "results.csv").exists()

    # checkpoint reload + val
    model2 = Model(str(run_dir / "best.pt"))
    assert model2.net.nc == 2
    val = model2.val(str(tiny_dataset), **COMMON)
    assert 0.0 <= val["mAP50"] <= 1.0

    # predict returns Results with original-image coordinates
    results = model2.predict(str(tiny_dataset.parent / "images" / "val"), conf=0.001, **COMMON)
    assert len(results) == 4
    for r in results:
        assert r.orig_shape == (96, 96)
        if len(r):
            assert r.boxes[:, [0, 2]].max() <= 96

    # resume continues from saved epoch
    result2 = model2.train(str(tiny_dataset), epochs=3, project=str(clean_runs), name="t",
                           resume=str(run_dir / "last.pt"), **COMMON)
    ckpt = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 2  # 0-indexed third epoch


def test_threshold_tuning_and_report(tiny_dataset, clean_runs):
    model = Model("trainer-n")
    model.train(str(tiny_dataset), epochs=1, project=str(clean_runs), name="r", **COMMON)
    thr = model.tune_thresholds(str(tiny_dataset), save_path=clean_runs / "thr.toml", **COMMON)
    assert set(thr["thresholds"]) == {"red", "green"}
    assert (clean_runs / "thr.toml").exists()

    from trainer.detection.config import load_config
    from trainer.detection.data import parse_dataset_toml
    from trainer.detection.reporting import generate_report

    result = model.val(str(tiny_dataset), **COMMON)
    parsed = parse_dataset_toml(tiny_dataset)
    report = generate_report(clean_runs / "r", result, parsed["names"],
                             load_config({"data": str(tiny_dataset)}), thresholds=thr)
    text = report.read_text()
    assert "<html" in text and "Per-class AP" in text and "red" in text


def test_finetune_nc_remap(tiny_dataset, clean_runs):
    model = Model("trainer-n")  # nc=80 fresh
    result = model.finetune(str(tiny_dataset), freeze=4, epochs=1,
                            project=str(clean_runs), name="ft", **COMMON)
    assert model.net.nc == 2
    assert "mAP50" in result
