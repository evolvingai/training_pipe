"""Regression fixtures for the feat/musgd correctness review."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from trainer.detection.config import load_config, load_toml
from trainer.detection.config.resume import resume_config
from trainer.detection.data.build import DistributedWeightedSampler
from trainer.detection.data.epoch_sampler import DATA_RNG_VERSION, EpochSampler
from trainer.detection.engine.musgd import MuSGD, newton_schulz
from trainer.detection.engine.trainer import apply_optimizer_step
from trainer.detection.engine.val_writer import write_cfg
from trainer.detection.engine.validator import validate
from trainer.detection.nn.loss import TaskAlignedAssigner
from trainer.detection.utils.ema import ModelEMA
from trainer.detection.utils.metrics import ConfusionMatrix, DetMetrics, PerImageStats, compute_ap, match_predictions
from trainer.detection.utils.nms import non_max_suppression


def test_overlapping_gt_falls_back_to_unmatched():
    gt = torch.tensor([[0., 0., 10., 10.], [2., 0., 12., 10.]])
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 0., 11., 10.]])
    assert match_predictions(boxes, torch.zeros(2), gt, torch.zeros(2))[:, 0].all()
    pred = torch.cat((boxes, torch.tensor([[.9, 0.], [.8, 0.]])), 1)
    cm = ConfusionMatrix(1, iou_thres=.5)
    cm.update(pred, gt, torch.zeros(2))
    assert cm.matrix[0, 0] == 2
    per = PerImageStats(1)
    per.update("image", pred, gt, torch.zeros(2))
    assert per._records[0]["tp"].tolist() == [2]


def test_ap_endpoints_and_equal_confidence_order():
    assert compute_ap(np.array([.5]), np.array([1.])) == pytest.approx(51 / 101)
    assert compute_ap(np.array([0.]), np.array([0.])) == 0
    metrics = DetMetrics(1)
    # At equal confidence, retain input order: FP before TP, AP=0.5.
    metrics.update(torch.tensor([[20., 20., 30., 30., .9, 0.], [0., 0., 10., 10., .9, 0.]]),
                   torch.tensor([[0., 0., 10., 10.]]), torch.zeros(1))
    assert metrics.compute()["mAP50-95"] == pytest.approx(.5)


@pytest.mark.parametrize("third_valid", [False, True])
def test_assignment_conflict_cannot_be_stolen(monkeypatch, third_valid):
    assigner = TaskAlignedAssigner(topk=1, num_classes=3)
    monkeypatch.setattr(assigner, "_pairwise_iou", lambda gt, pred:
                        torch.tensor([[[.6, 0.], [.7, 0.], [.99, 1.]]]))
    labels, _, _, fg = assigner(
        torch.full((1, 2, 3), .5), torch.ones(1, 2, 4),
        torch.tensor([[1., 1.], [2., 2.]]), torch.tensor([[[0.], [1.], [2.]]]),
        torch.tensor([[[0., 0., 3., 3.]]] * 3).transpose(0, 1),
        torch.tensor([[[1.], [1.], [float(third_valid)]]]),
    )
    assert fg[0, 0] and labels[0, 0] == 1


@pytest.mark.parametrize("nesterov", [False, True])
def test_musgd_decay_uses_original_parameter(nesterov):
    initial = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10
    p = initial.clone().requires_grad_()
    grad = torch.full_like(p, .3)
    p.grad = grad.clone()
    lr, momentum, wd, mw = .1, .9, .4, .2
    opt = MuSGD([p], lr=lr, momentum=momentum, weight_decay=wd,
                nesterov=nesterov, use_muon=True, muon_weight=mw)
    mu = grad * (1 - momentum)
    if nesterov:
        mu = mu * momentum + grad * (1 - momentum)
    sg = grad + wd * initial
    if nesterov:
        sg = sg * (1 + momentum)
    expected = initial - lr * mw * (4 / 3) ** .5 * newton_schulz(mu) - lr * sg
    opt.step()
    torch.testing.assert_close(p, expected, atol=1e-7, rtol=1e-6)
    assert torch.equal(p.grad, grad)


def test_zero_muon_skips_orthogonalization(monkeypatch):
    import trainer.detection.engine.musgd as module
    monkeypatch.setattr(module, "newton_schulz", lambda *args: pytest.fail("unnecessary NS call"))
    p = torch.ones(4, 3, requires_grad=True)
    ref = p.detach().clone().requires_grad_()
    kwargs = {"lr": .1, "momentum": .9, "weight_decay": .01, "nesterov": True}
    opt = MuSGD([p], use_muon=True, muon_weight=0, **kwargs)
    sgd = torch.optim.SGD([ref], **kwargs)
    for _ in range(5):
        p.grad, ref.grad = p.detach().clone(), ref.detach().clone()
        opt.step()
        sgd.step()
    assert "muon_buf" not in opt.state[p]
    torch.testing.assert_close(p, ref)


def test_musgd_channels_last_matches_contiguous():
    init = torch.randn(4, 3, 3, 3)
    a = init.clone().requires_grad_()
    b = init.clone().to(memory_format=torch.channels_last).requires_grad_()
    opts = [MuSGD([p], lr=.03, momentum=.9, weight_decay=.01, use_muon=True) for p in (a, b)]
    for _ in range(4):
        for p, opt in zip((a, b), opts):
            p.grad = p.detach().clone()
            opt.step()
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


class FakeScaler:
    def __init__(self):
        self.scale = 8.
        self.skip = True
    def unscale_(self, optimizer):
        pass
    def get_scale(self):
        return self.scale
    def step(self, optimizer):
        if not self.skip:
            optimizer.step()
    def update(self):
        if self.skip:
            self.scale /= 2


def test_amp_skip_does_not_advance_ema():
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    ema = ModelEMA(model)
    scaler = FakeScaler()
    original = copy.deepcopy(model.state_dict())
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    applied, _ = apply_optimizer_step(scaler, opt, model, ema)
    assert not applied and ema.step_count == 0
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
    scaler.skip = False
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    assert apply_optimizer_step(scaler, opt, model, ema)[0]
    assert ema.step_count == 1


def test_multilabel_nms_respects_per_class_thresholds():
    decoded = torch.tensor([[[5.], [5.], [4.], [4.], [.8], [.7]]])
    assert len(non_max_suppression(decoded)[0]) == 1
    assert len(non_max_suppression(decoded, multi_label=True)[0]) == 2
    pred = non_max_suppression(decoded, multi_label=True, per_class_conf=torch.tensor([.9, .6]))[0]
    assert pred[:, 5].tolist() == [1.]


def test_config_arrays_nested_values_roundtrip(tmp_path):
    cfg = load_config()
    cfg["extra_metadata"] = {"nested": {"text": "line one\nline two"}, "items": [{"x": 1}]}
    write_cfg(tmp_path, cfg, "missing.toml", "trainer-n")
    assert load_toml(tmp_path / "cfg" / "args.toml") == cfg


def test_unsafe_options_fail_early():
    for overrides in ({"label_smoothing": .1}, {"early_stop_metric": "val_loss"},
                      {"tune_metric": "val_loss"}, {"resume_mode": "unknown"}):
        with pytest.raises(ValueError):
            load_config(overrides)
    with pytest.raises(KeyError):
        load_config({"nms_multi_labl": True})


def test_resume_configuration_contract():
    saved = load_config({"optimizer": "musgd", "epochs": 400, "ema_decay": .9998})
    checkpoint = {"cfg": saved, "data_rng_version": DATA_RNG_VERSION,
                  "raw_model": {}, "optimizer": {}, "epoch": 9}
    assert resume_config(checkpoint, {"workers": 2})["optimizer"] == "musgd"
    assert resume_config(checkpoint, {})["ema_decay"] == .9998
    for changed in ({"optimizer": "sgd"}, {"epochs": 600}, {"batch": 32}):
        with pytest.raises(ValueError):
            resume_config(checkpoint, changed)
    assert resume_config(checkpoint, {"resume_mode": "continue", "epochs": 600})["epochs"] == 600
    checkpoint.pop("data_rng_version")
    with pytest.raises(ValueError, match="legacy"):
        resume_config(checkpoint, {})


def test_epoch_sampler_repeat_draws_and_closed_phase():
    base = DistributedWeightedSampler([100., 0.], 1, 0, seed=7)
    sampler = EpochSampler(base, 7, 0, epochs=4, close_mosaic=2)
    keys = list(sampler)
    assert keys[0][0] == keys[1][0] == 0 and keys[0][1] != keys[1][1]
    assert not keys[0][2] and keys == list(sampler)
    sampler.set_epoch(2)
    assert base.epoch == 2 and all(key[2] for key in sampler)
    assert next(iter(sampler))[1] != keys[0][1]


def test_distributed_epoch_sampler_shards_and_replays():
    from torch.utils.data import DistributedSampler
    samplers = [EpochSampler(DistributedSampler(range(8), num_replicas=2, rank=rank, seed=3),
                             seed=3, rank=rank, epochs=4, close_mosaic=1) for rank in (0, 1)]
    for sampler in samplers:
        sampler.set_epoch(3)
    a, b = [list(sampler) for sampler in samplers]
    assert {key[0] for key in a}.isdisjoint({key[0] for key in b})
    assert {key[0] for key in a + b} == set(range(8))
    assert all(key[2] for key in a + b)
    assert a == list(samplers[0]) and b == list(samplers[1])


class FixedDetector(torch.nn.Module):
    nc = 1
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
    def forward(self, imgs):
        # Original [8,8,12,12], resized by .5 and padded vertically by 8.
        return torch.tensor([7., 15., 6., 6., .9]).reshape(1, 5, 1).repeat(len(imgs), 1, 1), []


class FixedLoader:
    def __init__(self, annotations, batch_size):
        self.batch_size = batch_size
        self.dataset = SimpleNamespace(annotations_json=annotations,
                                       image_id_by_path={"a": 42, "b": 43}, category_id_for_class={0: 7})
    def __len__(self):
        return 2 // self.batch_size
    def __iter__(self):
        for offset in range(0, 2, self.batch_size):
            n = self.batch_size
            yield {"img": torch.zeros(n, 3, 32, 32, dtype=torch.uint8),
                   "batch_idx": torch.arange(n), "bboxes": torch.tensor([[7/32, 15/32, 6/32, 6/32]] * n),
                   "cls": torch.zeros(n), "paths": ["a", "b"][offset:offset+n],
                   "orig_shapes": [(32, 64)] * n, "pads": [(0, 8)] * n, "ratios": [.5] * n}


def test_comparison_batch_labels_and_box_rescaling(tmp_path, monkeypatch):
    from trainer.detection.data import coco_sota_fixes
    captured = []
    def evaluate(path, predictions, max_dets, image_ids):
        captured.extend(predictions)
        assert max_dets == [1, 10, 100] and image_ids == [42, 43]
        assert all(p["category_id"] == 7 and p["bbox"] == [8., 8., 12., 12.] for p in predictions)
        return {"AP": 1., "AP50": 1.}
    monkeypatch.setattr(coco_sota_fixes, "coco_evaluate", evaluate)
    cfg = load_config({"compare_pycoco": True})
    results = [validate(FixedDetector(), FixedLoader(tmp_path / "annotations.json", batch),
                        cfg, {0: "thing"}) for batch in (1, 2)]
    assert len(captured) == 4
    assert results[0]["mAP50-95"] == results[1]["mAP50-95"] == 1.
    assert results[0]["val_loss"] is None


def test_validation_restores_mode_on_error():
    model = FixedDetector().train()
    loader = SimpleNamespace(dataset=object())
    with pytest.raises(TypeError):
        validate(model, loader, load_config(), {0: "thing"})
    assert model.training


def test_export_sidecar_declares_external_nms(tmp_path):
    from trainer.detection.engine.exporter import write_postprocess_config
    path = write_postprocess_config(tmp_path / "model.onnx", load_config({"nms_multi_label": True}), {0: "x"})
    content = json.loads(path.read_text())
    assert content["nms_in_graph"] is False
    assert content["postprocess"]["nms_multi_label"] is True
