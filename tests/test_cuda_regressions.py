"""GPU-only release gates; CPU CI must not stand in for these checks."""

import pytest
import torch

from trainer.detection.engine.musgd import MuSGD, newton_schulz
from trainer.detection.engine.trainer import apply_optimizer_step
from trainer.detection.utils.ema import ModelEMA

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA release gate requires a GPU")


def test_cuda_newton_schulz_and_fp32_fallback(monkeypatch):
    matrix = torch.randn(32, 16, device="cuda")
    result = newton_schulz(matrix)
    assert result.shape == matrix.shape and torch.isfinite(result).all()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    # The mocked pre-Ampere device would not have TF32, but this test itself may
    # execute on Ampere/Ada where the process default permits TF32 matmuls.
    # Compare the fallback under the arithmetic it would actually use.
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        fallback = newton_schulz(matrix)
        torch.testing.assert_close(fallback.cpu(), newton_schulz(matrix.cpu()), atol=1e-4, rtol=1e-4)
    finally:
        torch.set_float32_matmul_precision(previous_precision)


def test_cuda_compiled_amp_step_and_overflow():
    raw = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
                              torch.nn.Conv2d(8, 2, 1)).cuda()
    model = torch.compile(raw)
    matrices = [p for p in model.parameters() if p.ndim >= 2]
    vectors = [p for p in model.parameters() if p.ndim < 2]
    optimizer = MuSGD([{"params": matrices, "use_muon": True}, {"params": vectors}],
                      lr=.01, momentum=.9, weight_decay=.001)
    ema = ModelEMA(model)
    scaler = torch.amp.GradScaler()
    x = torch.randn(2, 3, 32, 32, device="cuda")
    for overflow in (False, True, False):
        with torch.autocast("cuda"):
            loss = model(x).square().mean()
        scaler.scale(loss).backward()
        before = {k: v.clone() for k, v in raw.state_dict().items()}
        updates = ema.step_count
        if overflow:
            next(model.parameters()).grad.fill_(float("inf"))
        applied, _ = apply_optimizer_step(scaler, optimizer, model, ema)
        assert applied != overflow
        assert ema.step_count == updates + int(applied)
        if overflow:
            for key, value in raw.state_dict().items():
                assert torch.equal(value, before[key])
    assert all(torch.isfinite(p).all() for p in model.parameters())


def _teacher_checkpoint(path, names=None):
    from trainer.detection.nn.model import DetectionModel
    from trainer.detection.utils.checkpoints import atomic_torch_save
    model = DetectionModel("trainer-n", nc=2)
    atomic_torch_save({"model": model.state_dict(), "model_cfg": model.architecture_spec(), "nc": 2,
                      "ch": 3, "reg_max": 16, "names": names or {0: "a", 1: "b"}}, path)


def test_cuda_compiled_distillation_and_batched_qkv(tmp_path):
    from trainer.detection.config import load_config
    from trainer.detection.engine.distillation import Distiller
    from trainer.detection.engine.musgd import batched_newton_schulz
    from trainer.detection.nn.model import DetectionModel
    torch.manual_seed(0)
    path = tmp_path / "teacher.pt"
    _teacher_checkpoint(path)
    cfg = load_config({"teacher": str(path), "distill_feature_weight": 1., "distill_cls_weight": 1.,
                       "distill_box_weight": 1.})
    raw = DetectionModel("trainer-n", nc=2).cuda()
    kd = Distiller(raw, cfg, {0: "a", 1: "b"}, torch.device("cuda:0"))
    model = torch.compile(raw)
    optimizer = MuSGD([{"params": list(model.parameters())}, {"params": list(kd.adapters.parameters())}],
                      lr=.001, momentum=.9)
    # This is an integration test, not a dynamic-loss-scale test (covered
    # above).  Keep its first step finite and use the [0, 1] range seen by the
    # trainer after uint8 image normalization.
    ema, scaler = ModelEMA(model), torch.amp.GradScaler(init_scale=1024.)
    images = torch.rand(2, 3, 64, 64, device="cuda")
    batch = {"batch_idx": torch.tensor([0, 1]), "bboxes": torch.tensor([[.5, .5, .8, .8]] * 2)}
    with torch.autocast("cuda"):
        maps, features = model(images, return_features=True)
        loss, _ = kd.loss(maps, features, kd.teacher_forward(images), batch)
    scaler.scale(loss).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for group in optimizer.param_groups for p in group["params"])
    applied, _ = apply_optimizer_step(scaler, optimizer, model, ema)
    assert applied and ema.step_count == 1
    assert all(p.grad is None for p in kd.teacher.parameters())
    matrices = torch.randn(3, 16, 8, device="cuda")
    actual = batched_newton_schulz(matrices, steps=3)
    expected = torch.stack([newton_schulz(m, steps=3) for m in matrices])
    torch.testing.assert_close(actual, expected, atol=.06, rtol=.06)  # BF16 batched GEMM tolerance


def _ddp_distillation_worker(rank, rendezvous, teacher_path):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from trainer.detection.config import load_config
    from trainer.detection.engine.distillation import Distiller
    from trainer.detection.nn.model import DetectionModel
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        cfg = load_config({"teacher": str(teacher_path), "distill_feature_weight": 1.,
                           "distill_cls_weight": 1., "distill_box_weight": 1.})
        raw = DetectionModel("trainer-n", nc=2).to(device)
        kd = Distiller(raw, cfg, {0: "a", 1: "b"}, device)
        model = DDP(raw, device_ids=[rank])
        kd.adapters = DDP(kd.adapters, device_ids=[rank])
        params = list(model.parameters()) + list(kd.adapters.parameters())
        optimizer = torch.optim.SGD(params, lr=.001)
        torch.manual_seed(rank + 20)
        for _ in range(2):
            images = torch.randn(2, 3, 64, 64, device=device)
            batch = {"batch_idx": torch.tensor([0, 1]), "bboxes": torch.tensor([[.5, .5, .8, .8]] * 2)}
            maps, features = model(images, return_features=True)
            loss, _ = kd.loss(maps, features, kd.teacher_forward(images), batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        for p in params:
            other = p.detach().clone()
            dist.broadcast(other, src=0)
            torch.testing.assert_close(p, other, atol=0, rtol=0)
        assert all(p.grad is None for p in kd.teacher.parameters())
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_two_gpu_distillation_adapter_reduction(tmp_path):
    import torch.multiprocessing as mp
    teacher = tmp_path / "teacher.pt"
    _teacher_checkpoint(teacher)
    mp.spawn(_ddp_distillation_worker, args=(str(tmp_path / "rendezvous"), str(teacher)), nprocs=2, join=True)


def _ddp_resume_worker(rank, rendezvous, teacher_path, data_path, project):
    import os
    from pathlib import Path

    import torch.distributed as dist

    from trainer.detection.engine.trainer import Trainer
    from trainer.detection.utils.checkpoints import load_checkpoint
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        cfg = {"model": "trainer-n", "data": data_path, "project": project, "name": "ddp_resume",
               "teacher": teacher_path, "distill_feature_weight": 1., "distill_cls_weight": .5,
               "optimizer": "musgd", "muon_weight": .1, "imgsz": 64, "epochs": 2,
               "batch": 2, "nbs": 4, "workers": 0, "device": f"cuda:{rank}", "amp": False,
               "compile": False, "deterministic": True, "warmup_epochs": 0., "save_period": 1,
               "cache": "none", "close_mosaic": 1, "patience": 100}
        Trainer(overrides=cfg).train()
        weights = Path(project) / "ddp_resume/weights"
        reference = load_checkpoint(weights / "last.pt") if rank == 0 else None
        dist.barrier()
        Trainer(overrides={"resume": str(weights / "epoch_1.pt"), "device": f"cuda:{rank}"}).train()
        if rank == 0:
            actual = load_checkpoint(weights / "last.pt")
            def compare(left, right):
                if isinstance(left, torch.Tensor):
                    torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
                elif isinstance(left, dict):
                    assert left.keys() == right.keys()
                    for key in left:
                        compare(left[key], right[key])
                elif isinstance(left, (list, tuple)):
                    for a, b in zip(left, right):
                        compare(a, b)
                else:
                    assert left == right
            for key in ("raw_model", "model", "distillation", "optimizer", "ema_meta", "global_step"):
                compare(reference[key], actual[key])
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_two_gpu_distillation_trainer_resume(tmp_path, tiny_dataset):
    import torch.multiprocessing as mp
    teacher = tmp_path / "teacher.pt"
    _teacher_checkpoint(teacher, {0: "red", 1: "green"})
    mp.spawn(_ddp_resume_worker,
             args=(str(tmp_path / "rendezvous"), str(teacher), str(tiny_dataset), str(tmp_path)),
             nprocs=2, join=True)
