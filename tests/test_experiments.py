"""Experimental policies are opt-in and must preserve resume/inference contracts."""
import copy
import json

import pytest
import torch

from trainer.detection.api import Model
from trainer.detection.config import load_config
from trainer.detection.engine.distillation import Distiller, foreground_mask
from trainer.detection.engine.experiments import optimizer_partition
from trainer.detection.engine.musgd import MuSGD, batched_newton_schulz, muon_weight_at_epoch, newton_schulz
from trainer.detection.nn.loss import DetectionLoss, TaskAlignedAssigner
from trainer.detection.nn.model import DetectionModel
from trainer.detection.utils.boxes import bbox_iou
from trainer.detection.utils.checkpoints import CHECKPOINT_VERSION, atomic_torch_save, load_checkpoint, model_from_checkpoint

torch.set_num_threads(1)
COMMON = {"imgsz": 64, "batch": 4, "nbs": 4, "workers": 0, "device": "cpu", "amp": False, "compile": False,
          "optimizer": "musgd", "muon_weight": .1, "warmup_epochs": 0., "patience": 100, "deterministic": True,
          "mosaic": .8, "close_mosaic": 2, "cache": "none", "epochs": 4, "save_period": 1, "lr0": .01}


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal_state(left, right)
    else:
        assert a == b


@pytest.fixture()
def teacher_checkpoint(tmp_path):
    teacher = DetectionModel("trainer-s", nc=2)
    path = tmp_path / "teacher.pt"
    atomic_torch_save({"checkpoint_version": CHECKPOINT_VERSION, "model": teacher.state_dict(),
                      "model_cfg": teacher.architecture_spec(), "nc": 2, "ch": 3, "reg_max": 16,
                      "names": {0: "red", 1: "green"}, "epoch": 0, "cfg": load_config()}, path)
    return path


@pytest.mark.parametrize("changes", [
    {"muon_beta": 1.}, {"muon_ns_steps": 0}, {"muon_log_interval": -1}, {"reg_channels": -1},
    {"assignment_metric": "other"}, {"distill_cls_weight": 1.}, {"teacher": "teacher.pt"},
    {"teacher": "t.pt", "distill_feature_weight": 1., "autobatch": True},
    {"muon_schedule": "linear", "optimizer": "sgd"},
    {"muon_schedule": "linear", "optimizer": "musgd", "epochs": 100},
])
def test_experiment_config_rejects_invalid(changes):
    with pytest.raises(ValueError):
        load_config(changes)


def test_routing_coverage_and_exact_exclusions():
    model = DetectionModel("trainer-n")
    for option, reason in (("muon_exclude_outputs", "output"), ("muon_exclude_stem", "stem"),
                           ("muon_exclude_depthwise", "depthwise"), ("muon_split_qkv", "qkv_slices")):
        parts = optimizer_partition(model, load_config({"optimizer": "musgd", option: True}))
        groups, records = parts[:5], parts[-1]
        params = [p for group in groups for p in group]
        assert len(params) == len(set(map(id, params)))
        assert {id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad}
        changed = [r for r in records if r["reason"] == reason]
        assert changed
        assert all(r["route"] == ("muon+sgd" if reason == "qkv_slices" else "sgd") for r in changed)
        if reason == "output":
            assert len(changed) == model.head.nl * 2
        if reason == "stem":
            assert len(changed) == 1
        if reason == "qkv_slices":
            assert all(r["qkv_splits"] == 3 for r in changed)


@pytest.mark.parametrize("shape", [(4, 3), (3, 9), (4, 4)])
@pytest.mark.parametrize("steps", [3, 5])
def test_batched_ns_matches_individual(shape, steps):
    matrices = torch.randn(5, *shape)
    matrices[0].zero_()
    expected = torch.stack([newton_schulz(x, steps=steps) for x in matrices])
    torch.testing.assert_close(batched_newton_schulz(matrices, steps), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("batch_ns", [False, True])
def test_qkv_slicing_matches_three_independent_matrices(batch_ns):
    parameter = torch.zeros(12, 4, 1, 1, requires_grad=True)
    grad = torch.arange(48).float().reshape_as(parameter) / 50
    parameter.grad = grad
    opt = MuSGD([{"params": [parameter], "muon_splits": 3}], lr=1, momentum=0,
                use_muon=True, muon_weight=1, sgd_weight=0, batch_ns=batch_ns)
    opt.step()
    expected = -torch.cat([newton_schulz(x.reshape(4, 4)) for x in grad.chunk(3)], dim=0)
    # Batched GEMM can choose a different reduction kernel from three individual
    # GEMMs (notably with newer CUDA/PyTorch builds). This experiment promises
    # numerical tolerance, not bitwise slice equality.
    torch.testing.assert_close(parameter.flatten(1), expected, atol=1e-5, rtol=1e-4)


def test_independent_beta_diagnostics_and_state_roundtrip():
    parameter = torch.ones(4, 3, requires_grad=True)
    optimizer = MuSGD([{"params": [parameter], "param_names": ["hidden.weight"]}], lr=.01,
                      momentum=.8, muon_beta=.95, use_muon=True, weight_decay=.01)
    optimizer.collect_diagnostics = True
    for momentum in (.8, .937):
        optimizer.param_groups[0]["momentum"] = momentum
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    assert torch.allclose(optimizer.state[parameter]["muon_buf"], torch.full_like(parameter, .0975))
    row = optimizer.last_diagnostics[0]
    assert row["parameter"] == "hidden.weight" and row["muon_beta"] == .95
    assert row["muon_update_norm"] > 0 and row["sgd_update_norm"] > 0 and -1.00001 <= row["cosine"] <= 1.00001
    clone = parameter.detach().clone().requires_grad_()
    restored = MuSGD([clone])
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for p in (clone, parameter):
        p.grad = torch.full_like(p, .3)
    restored.step()
    optimizer.step()
    torch.testing.assert_close(clone, parameter, atol=0, rtol=0)


def test_taper_boundaries_and_zero_buffer_policy():
    cfg = load_config({"optimizer": "musgd", "epochs": 400, "muon_schedule": "linear", "muon_weight": .1})
    assert [muon_weight_at_epoch(cfg, e) for e in (0, 299, 300, 399, 400)] == [.1, .1, .1, 0., 0.]
    assert muon_weight_at_epoch(cfg, 350) == pytest.approx(.1 * 49 / 99)
    parameter = torch.ones(4, 3, requires_grad=True)
    optimizer = MuSGD([parameter], use_muon=True, momentum=.9)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    before = optimizer.state[parameter]["muon_buf"].clone()
    optimizer.param_groups[0]["muon_weight"] = 0.
    optimizer.step()
    assert torch.equal(before, optimizer.state[parameter]["muon_buf"])


def test_ciou_assignment_uses_real_ciou():
    gt = torch.tensor([[[0., 0., 5., 10.]]])
    pred = torch.tensor([[[1., 0., 7., 8.], [0., 0., 5., 10.]]])
    scores = torch.tensor([[[.9], [.1]]])
    anchor = torch.tensor([[2., 2.], [3., 3.]])
    iou = TaskAlignedAssigner(topk=2, num_classes=1)
    ciou = TaskAlignedAssigner(topk=2, num_classes=1, metric="ciou")
    args = (scores, pred, anchor, torch.zeros(1, 1, 1), gt, torch.ones(1, 1, 1))
    expected = bbox_iou(gt.unsqueeze(2), pred.unsqueeze(1), ciou=True)
    assert not torch.allclose(expected, iou._pairwise_iou(gt, pred))
    assert not torch.allclose(iou(*args)[2], ciou(*args)[2])
    assert torch.isfinite(ciou(*args)[2]).all()


def test_fixed_width_checkpoint_and_clipping_diagnostics():
    models = [DetectionModel("trainer-n", nc=2, reg_max=bins, reg_channels=64) for bins in (16, 8)]
    assert all(model.head.reg_channels == 64 for model in models)
    for model in models:
        checkpoint = {"model_cfg": model.architecture_spec(), "nc": 2, "ch": 3, "reg_max": model.head.reg_max}
        rebuilt = model_from_checkpoint(checkpoint)
        rebuilt.load_state_dict(model.state_dict())
        assert rebuilt.head.reg_channels == 64
    model = DetectionModel("trainer-n", nc=2, reg_max=2, reg_channels=64)
    loss_fn = DetectionLoss(model, load_config({"loss_diagnostics": True}))
    batch = {"batch_idx": torch.tensor([0, 1]), "cls": torch.tensor([0., 1.]),
             "bboxes": torch.tensor([[.5, .5, .99, .99], [.5, .5, .99, .99]])}
    loss, _ = loss_fn(model(torch.randn(2, 3, 64, 64)), batch)
    loss.backward()
    assert loss_fn.last_diagnostics["total_dfl_sides"] > 0
    assert loss_fn.last_diagnostics["clipped_dfl_sides"] > 0


@pytest.mark.parametrize("term", ["feature", "cls", "box"])
def test_distillation_gradients_teacher_frozen_and_empty_gt(teacher_checkpoint, term):
    student = DetectionModel("trainer-n", nc=2)
    cfg = load_config({"teacher": str(teacher_checkpoint), f"distill_{term}_weight": 1.})
    distiller = Distiller(student, cfg, {0: "red", 1: "green"}, torch.device("cpu"))
    before = {name: value.clone() for name, value in distiller.teacher.state_dict().items()}
    images = torch.randn(2, 3, 64, 64)
    maps, features = student(images, return_features=True)
    target = distiller.teacher_forward(images)
    batch = {"batch_idx": torch.tensor([0, 1]), "bboxes": torch.tensor([[.5, .5, .8, .8]] * 2)}
    kd_loss, terms = distiller.loss(maps, features, target, batch)
    assert kd_loss > 0 and terms[term] > 0
    kd_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
    assert all(p.grad is None and not p.requires_grad for p in distiller.teacher.parameters())
    equal_state(before, distiller.teacher.state_dict())
    if distiller.adapters is not None:
        assert all(p.grad is not None for p in distiller.adapters.parameters())
    empty = {"batch_idx": torch.empty(0, dtype=torch.long), "bboxes": torch.empty(0, 4)}
    assert not foreground_mask(empty, maps[0].shape, torch.device("cpu")).any()
    maps, features = student(images, return_features=True)
    empty_loss, _ = distiller.loss(maps, features, target, empty)
    assert empty_loss.item() == 0
    empty_loss.backward()


def test_teacher_class_or_box_mismatch_rejected(teacher_checkpoint):
    student = DetectionModel("trainer-n", nc=2, reg_max=8)
    cfg = load_config({"teacher": str(teacher_checkpoint), "distill_box_weight": 1.})
    with pytest.raises(ValueError, match="names/order"):
        Distiller(student, cfg, {0: "green", 1: "red"}, torch.device("cpu"))
    with pytest.raises(ValueError, match="reg_max"):
        Distiller(student, cfg, {0: "red", 1: "green"}, torch.device("cpu"))


def test_distillation_resume_and_inference_artifacts(teacher_checkpoint, tiny_dataset, tmp_path):
    options = {**COMMON, "teacher": str(teacher_checkpoint), "distill_feature_weight": 1.,
               "distill_cls_weight": .5, "distill_box_weight": .5, "profile": True,
               "muon_log_interval": 1, "loss_diagnostics": True}
    model = Model("trainer-n")
    model.train(str(tiny_dataset), project=str(tmp_path), name="kd", **options)
    weights = model.last_run_dir / "weights"
    reference = load_checkpoint(weights / "last.pt")
    assert reference["distillation"]["adapters"]
    assert all(not k.startswith("distill") for k in reference["raw_model"])
    selected = load_checkpoint(weights / "selected.pt")
    assert "distillation" not in selected
    Model(str(weights / "selected.pt"))  # deployment requires neither teacher nor adapters
    routing = json.loads((model.last_run_dir / "cfg/optimizer_routing.json").read_text())
    assert all(row["route"] == "sgd" for row in routing if row["name"].startswith("distill."))
    profile = json.loads((model.last_run_dir / "logs/profile.jsonl").read_text().splitlines()[0])
    assert all(profile[key] > 0 for key in ("teacher_forward", "optimizer_step", "backward", "ema"))
    assert (model.last_run_dir / "logs/muon_updates.jsonl").stat().st_size > 0
    resumed = Model(str(weights / "epoch_2.pt"))
    resumed.train(str(tiny_dataset), resume=str(weights / "epoch_2.pt"))
    actual = load_checkpoint(weights / "last.pt")
    for key in ("raw_model", "model", "distillation", "optimizer", "scheduler", "best_model", "global_step"):
        equal_state(reference[key], actual[key])
    # A teacher change must fail, never silently alter resumed targets.
    torch.save({"changed": True}, teacher_checkpoint)
    with pytest.raises((ValueError, KeyError)):
        resumed.train(str(tiny_dataset), resume=str(weights / "epoch_2.pt"))
    # Even with an unusable external teacher, the inference checkpoint still evaluates.
    deployed = Model(str(weights / "selected.pt"))
    deployed.val(str(tiny_dataset), workers=0, batch=2, imgsz=64, cache="none",
                 project=str(tmp_path), name="student_without_teacher")


def test_constant_branch_equivalence_and_taper_parent_preserved(tiny_dataset, tmp_path):
    model = Model("trainer-n")
    model.train(str(tiny_dataset), project=str(tmp_path), name="prefix", **COMMON)
    weights = model.last_run_dir / "weights"
    reference = load_checkpoint(weights / "last.pt")
    parent = weights / "epoch_2.pt"
    from trainer.detection.utils.provenance import file_sha256
    parent_hash = file_sha256(parent)
    constant = Model(str(parent))
    constant.train(str(tiny_dataset), resume=str(parent), resume_mode="branch", name="constant")
    actual = load_checkpoint(constant.last_run_dir / "weights/last.pt")
    for key in ("raw_model", "model", "optimizer", "scheduler", "best_model", "best_epoch"):
        equal_state(reference[key], actual[key])
    taper = Model(str(parent))
    taper.train(str(tiny_dataset), resume=str(parent), resume_mode="branch", name="taper",
                muon_schedule="linear", muon_taper_start=2, muon_taper_end=4)
    tapered = load_checkpoint(taper.last_run_dir / "weights/last.pt")
    assert all(g["muon_weight"] == 0 for g in tapered["optimizer"]["param_groups"])
    assert file_sha256(parent) == parent_hash
    assert not torch.equal(next(iter(reference["raw_model"].values())), next(iter(tapered["raw_model"].values())))


def test_raw_vs_ema_evaluation(teacher_checkpoint, tiny_dataset, tmp_path):
    checkpoint = load_checkpoint(teacher_checkpoint)
    checkpoint["raw_model"] = {k: v.clone() for k, v in checkpoint["model"].items()}
    name = next(k for k, v in checkpoint["raw_model"].items() if v.is_floating_point())
    checkpoint["raw_model"][name] += .1
    atomic_torch_save(checkpoint, teacher_checkpoint)
    model = Model(str(teacher_checkpoint))
    for selection, expected in (("raw", checkpoint["raw_model"]), ("ema", checkpoint["model"])):
        result = model.val(str(tiny_dataset), eval_weights=selection, device="cpu", workers=0, batch=2, imgsz=64,
                           project=str(tmp_path), name=selection, cache="none")
        equal_state(model.net.state_dict(), expected)
        assert result["checkpoint"]["selection"].lower() == selection


def test_end_to_end_checkpoint_benchmark(tiny_dataset):
    from trainer.detection.data import parse_dataset_toml
    from trainer.detection.engine.benchmark import benchmark_checkpoint
    parsed = parse_dataset_toml(tiny_dataset)
    paths = [str(path) for path in (parsed["root"] / "images/val").glob("*.jpg")]
    model = DetectionModel("trainer-n", nc=2).eval()
    cfg = load_config({"imgsz": 64, "batch": 1, "val_conf": .25})
    result = benchmark_checkpoint(model, paths, cfg, torch.device("cpu"), warmup=0, iterations=2)
    assert result["parameters"] == model.num_params()
    assert result["batch"] == 1 and result["throughput_images_s"] > 0
    assert all(result["timings"][key]["p50_ms"] > 0 for key in ("forward", "nms", "end_to_end", "read_preprocess"))
