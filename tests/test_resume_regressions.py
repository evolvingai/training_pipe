import json

import pytest
import torch

from trainer.detection.api import Model
from trainer.detection.config import load_config
from trainer.detection.data import build_dataloader, parse_dataset_toml
from trainer.detection.utils.checkpoints import load_checkpoint
from trainer.detection.utils.metrics import DetMetrics

COMMON = {"imgsz": 64, "batch": 4, "nbs": 4, "workers": 0, "device": "cpu", "amp": False,
          "compile": False, "optimizer": "sgd", "warmup_epochs": .5, "patience": 100,
          "deterministic": True, "mosaic": .8, "close_mosaic": 2, "cache": "none", "epochs": 4,
          "save_period": 1, "ema_decay": .9998, "ema_warmup_steps": 7, "lr0": .01}


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("optimizer", ["sgd", "adamw", "musgd"])
def test_resume_before_at_after_mosaic_closure(optimizer, tiny_dataset, tmp_path):
    options = dict(COMMON, optimizer=optimizer)
    original = Model("trainer-n")
    original.train(str(tiny_dataset), project=str(tmp_path), name=optimizer, **options)
    weights = original.last_run_dir / "weights"
    reference = load_checkpoint(weights / "last.pt")
    assert reference["ema_meta"]["decay_max"] == .9998
    if optimizer != "adamw":
        assert reference["optimizer"]["param_groups"][0]["momentum"] == .937
    for saved_epoch in (1, 2, 3):
        checkpoint = weights / f"epoch_{saved_epoch}.pt"
        resumed = Model(str(checkpoint))
        # Recipe comes from the checkpoint; no need to restate every option.
        resumed.train(str(tiny_dataset), resume=str(checkpoint))
        actual = load_checkpoint(resumed.last_run_dir / "weights" / "last.pt")
        for key in ("raw_model", "model", "optimizer", "scheduler", "ema_meta", "global_step",
                    "attempted_steps", "batch_step", "best_model", "best_epoch"):
            assert_state_equal(reference[key], actual[key])


@pytest.mark.parametrize("balanced", [False, True])
def test_persistent_workers_replay_epochs_and_augmentations(tiny_dataset, balanced):
    data = parse_dataset_toml(tiny_dataset)
    cfg = load_config(dict(COMMON, workers=2, balanced_sampling=balanced))
    uninterrupted = build_dataloader(data, cfg)
    uninterrupted.timeout = 30
    # Visit the preceding epoch, leaving persistent workers alive.
    uninterrupted.sampler.set_epoch(0)
    list(uninterrupted)
    for epoch in (1, 2, 3):
        uninterrupted.sampler.set_epoch(epoch)
        expected = list(uninterrupted)
        fresh = build_dataloader(data, load_config(dict(cfg, workers=0)))
        fresh.sampler.set_epoch(epoch)
        actual = list(fresh)
        assert len(expected) == len(actual)
        for a, b in zip(expected, actual):
            assert a["paths"] == b["paths"]
            for key in ("img", "cls", "bboxes", "batch_idx"):
                assert torch.equal(a[key], b[key]), (epoch, key)


@pytest.mark.parametrize("restore_best", [False, True])
def test_final_metrics_weights_and_artifacts_agree(monkeypatch, tiny_dataset, tmp_path, restore_best):
    import trainer.detection.engine.trainer as module
    states = []
    def fake_validate(model, *args, **kwargs):
        current = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if len(states) < 2:
            states.append(current)
            score = .8 if len(states) == 1 else .2
        else:
            chosen = 0 if restore_best else 1
            assert_state_equal(current, states[chosen])
            score = .8 if restore_best else .2
        result = DetMetrics(2).compute()
        result.update({"mAP50-95": score, "mAP50": score, "val_loss": None})
        return result
    monkeypatch.setattr(module, "validate", fake_validate)
    model = Model("trainer-n")
    result = model.train(str(tiny_dataset), project=str(tmp_path), name="selected",
                         **dict(COMMON, epochs=2, early_stop_restore_best=restore_best))
    assert result["mAP50-95"] == (.8 if restore_best else .2)
    assert result["checkpoint"]["epoch"] == (1 if restore_best else 2)
    selected = load_checkpoint(result["checkpoint"]["path"])
    assert "optimizer" not in selected  # never attach mismatched optimizer state
    assert_state_equal(selected["model"], model.net.state_dict())
    run = model.last_run_dir
    logged = json.loads((run / "logs" / "val_results.json").read_text())
    assert logged["checkpoint"] == result["checkpoint"]
    assert logged["global"]["val_loss"] is None
    assert (run / "cfg" / "provenance.json").exists()
    assert "weights/last.pt" in json.loads((run / "manifest.json").read_text())["files"]
    assert (run / "plots" / "curve_mAP.png").exists()


def test_continuation_first_lr_matches_new_schedule(tiny_dataset, tmp_path, monkeypatch):
    import trainer.detection.engine.trainer as module
    model = Model("trainer-n")
    options = dict(COMMON, epochs=2, warmup_epochs=0.)
    model.train(str(tiny_dataset), project=str(tmp_path), name="schedule", **options)
    checkpoint = model.last_run_dir / "weights" / "last.pt"
    seen_lrs = []
    actual_step = module.apply_optimizer_step
    def capture(scaler, optimizer, model, ema):
        seen_lrs.append(optimizer.param_groups[-1]["lr"])
        return actual_step(scaler, optimizer, model, ema)
    monkeypatch.setattr(module, "apply_optimizer_step", capture)
    model.train(str(tiny_dataset), resume=str(checkpoint), resume_mode="continue", epochs=4, lr0=.02)
    cfg = load_config(dict(options, epochs=4, lr0=.02))
    import math
    expected = .02 * (cfg["lrf"] + (1-cfg["lrf"]) * .5 * (1 + math.cos(math.pi * 2/4)))
    assert seen_lrs[0] == pytest.approx(expected)
    assert model.last_run_dir != checkpoint.parent.parent
    assert load_checkpoint(checkpoint)["epoch"] == 1  # parent preserved
