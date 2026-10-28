"""Integration smoke: 2-epoch CPU training, resume, val, predict, thresholds, report."""

import torch
import torch.nn as nn

from trainer.detection.api import Model
from trainer.detection.config import load_config
from trainer.detection.engine.trainer import Trainer, accumulation_divisor
from trainer.detection.nn.model import DetectionModel

COMMON = dict(imgsz=96, batch=4, workers=0, device="cpu", amp=False, mosaic=0.0,
              warmup_epochs=0.5, patience=100, deterministic=True, compile=False)


def test_partial_accumulation_window_uses_its_actual_size():
    assert [accumulation_divisor(i, 10, 4) for i in range(10)] == [4] * 8 + [2, 2]
    assert [accumulation_divisor(i, 8, 4) for i in range(8)] == [4] * 8


def test_frozen_batchnorm_stays_in_eval_mode(tmp_path):
    model = DetectionModel("trainer-n")
    trainer = Trainer(
        cfg=load_config({"freeze": 2, "project": str(tmp_path), "name": "freeze"}), model=model,
    )
    trainer._freeze(model)
    model.train()
    trainer._keep_frozen_modules_eval()
    frozen_bn = [m for layer in model.layers[:2] for m in layer.modules() if isinstance(m, nn.BatchNorm2d)]
    assert frozen_bn and all(not module.training for module in frozen_bn)


def test_train_val_predict_cycle(tiny_dataset, clean_runs):
    model = Model("trainer-n")
    result = model.train(str(tiny_dataset), epochs=2, project=str(clean_runs), name="t", **COMMON)
    assert "mAP50" in result
    run_dir = clean_runs / "t"
    assert (run_dir / "weights" / "best.pt").exists()
    assert (run_dir / "weights" / "last.pt").exists()
    assert (run_dir / "cfg" / "args.toml").exists()
    assert (run_dir / "logs" / "results.csv").exists()

    # checkpoint reload + val
    model2 = Model(str(run_dir / "weights" / "best.pt"))
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
    model2.train(str(tiny_dataset), epochs=3, project=str(clean_runs), name="t",
                 resume=str(run_dir / "weights" / "last.pt"), resume_mode="continue", **COMMON)
    ckpt = torch.load(model2.last_run_dir / "weights" / "last.pt", map_location="cpu", weights_only=True)
    assert ckpt["epoch"] == 2  # 0-indexed third epoch
    assert ckpt["scheduler"]["last_epoch"] == 3


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
    assert text.startswith("# Evaluation Report") and "Per-class Metrics" in text and "red" in text


def test_finetune_nc_remap(tiny_dataset, clean_runs):
    model = Model("trainer-n")  # nc=80 fresh
    result = model.finetune(str(tiny_dataset), freeze=4, epochs=1,
                            project=str(clean_runs), name="ft", **COMMON)
    assert model.net.nc == 2
    assert "mAP50" in result


def test_musgd_trains_and_resumes_bit_exactly(tiny_dataset, clean_runs):
    """MuSGD carries two momentum buffers per orthogonalized weight; resume must
    restore both, so run the bit-exact resume check against it as well."""
    musgd = dict(COMMON, optimizer="musgd", lr0=0.02, muon_weight=0.2, sgd_weight=1.0)
    model = Model("trainer-n")
    model.train(str(tiny_dataset), epochs=2, save_period=1,
                project=str(clean_runs), name="musgd", **musgd)
    weights = clean_runs / "musgd" / "weights"
    assert (weights / "best.pt").exists()
    uninterrupted = torch.load(weights / "last.pt", map_location="cpu", weights_only=True)
    first_epoch = weights / "epoch_1.pt"

    resumed_model = Model(str(first_epoch))
    resumed_model.train(str(tiny_dataset), epochs=2, save_period=1,
                        project=str(clean_runs), name="musgd", resume=str(first_epoch), **musgd)
    resumed = torch.load(weights / "last.pt", map_location="cpu", weights_only=True)
    for key, value in uninterrupted["raw_model"].items():
        assert torch.equal(resumed["raw_model"][key], value), key


def test_epoch_checkpoint_resume_is_bit_exact(tiny_dataset, clean_runs):
    model = Model("trainer-n")
    model.train(
        str(tiny_dataset), epochs=2, save_period=1, project=str(clean_runs), name="exact",
        fliplr=0.5, hsv_h=0.02, **COMMON,
    )
    weights = clean_runs / "exact" / "weights"
    uninterrupted = torch.load(weights / "last.pt", map_location="cpu", weights_only=True)
    first_epoch = weights / "epoch_1.pt"

    resumed_model = Model(str(first_epoch))
    resumed_model.train(
        str(tiny_dataset), epochs=2, save_period=1, project=str(clean_runs), name="exact",
        resume=str(first_epoch), fliplr=0.5, hsv_h=0.02, **COMMON,
    )
    resumed = torch.load(weights / "last.pt", map_location="cpu", weights_only=True)
    assert resumed["global_step"] == uninterrupted["global_step"]
    assert resumed["scheduler"] == uninterrupted["scheduler"]
    for key, value in uninterrupted["raw_model"].items():
        assert torch.equal(resumed["raw_model"][key], value), key
