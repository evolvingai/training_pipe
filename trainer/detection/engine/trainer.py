"""Training loop: single GPU or torchrun DDP, AMP, EMA, exact resume, and tracking."""

from __future__ import annotations

import logging
import math
import os
import random
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from trainer.detection.config import load_config, merge_overrides
from trainer.detection.data import build_dataloader, parse_dataset_toml
from trainer.detection.engine.musgd import MuSGD, muon_weight_at_epoch
from trainer.detection.engine.experiments import StepProfiler, append_jsonl, optimizer_partition
from trainer.detection.engine.validator import comparison_metrics, per_class_metrics, per_class_table, validate
from trainer.detection.nn.loss import DetectionLoss
from trainer.detection.nn.model import DetectionModel
from trainer.detection.trackers import MultiTracker, build_tracker
from trainer.detection.utils.checks import autobatch, cuda_device_index, init_seeds, select_device
from trainer.detection.utils.checkpoints import (
    CHECKPOINT_VERSION,
    atomic_torch_save,
    checkpoint_safe,
    load_checkpoint,
    model_from_checkpoint,
)
from trainer.detection.utils.ema import ModelEMA
from trainer.detection.utils.logging import LOGGER, format_seconds, increment_path
from trainer.detection.data.epoch_sampler import DATA_RNG_VERSION
from trainer.detection.utils.provenance import collect_provenance, file_sha256, write_json, write_provenance

HEADER = (f"{'Epoch':>9s} {'box':>8s} {'cls':>8s} {'dfl':>8s} {'mAP50':>7s} {'mAP50-95':>9s} "
          f"{'P':>6s} {'R':>6s} {'LR':>9s} {'data%':>6s} {'time':>8s}")


def accumulation_divisor(batch_index: int, num_batches: int, accumulate: int) -> int:
    """Return the true window size, including a shorter final accumulation window."""
    remainder = num_batches % accumulate
    if remainder and batch_index >= num_batches - remainder:
        return remainder
    return accumulate


def apply_optimizer_step(scaler, optimizer, model, ema, profiler=None) -> tuple[bool, float]:
    """Update EMA only when GradScaler applies the optimizer update."""
    measure = profiler.measure if profiler is not None else lambda _: nullcontext()
    with measure("unscale_clip"):
        scaler.unscale_(optimizer)
        # Includes training-only adapters, which are not part of the EMA model.
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
    before = scaler.get_scale()
    with measure("optimizer_step"):
        scaler.step(optimizer)
    with measure("scaler_update"):
        scaler.update()
    applied = scaler.get_scale() >= before
    optimizer.zero_grad(set_to_none=True)
    if applied:
        with measure("ema"):
            ema.update(model)
    return applied, float(norm)


def _rng_state() -> dict:
    np_state = np.random.get_state()
    return {
        "python": checkpoint_safe(random.getstate()),
        "numpy": {
            "bit_generator": np_state[0],
            "keys": torch.from_numpy(np_state[1].copy()),
            "pos": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _nested_tuple(value):
    return tuple(_nested_tuple(v) for v in value) if isinstance(value, list) else value


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(_nested_tuple(state["python"]))
    np_state = state["numpy"]
    np.random.set_state((
        str(np_state["bit_generator"]), np_state["keys"].cpu().numpy(), int(np_state["pos"]),
        int(np_state["has_gauss"]), float(np_state["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


class EarlyStopper:
    def __init__(self, cfg: dict):
        self.patience = int(cfg["patience"])
        self.min_delta = float(cfg["early_stop_min_delta"])
        self.min_epochs = int(cfg["early_stop_min_epochs"])
        self.best = -math.inf
        self.best_epoch = 0
        self.counter = 0

    def step(self, value: float, epoch: int) -> bool:
        if value > self.best + self.min_delta:
            self.best = value
            self.best_epoch = epoch
            self.counter = 0
            return False
        self.counter += 1
        return epoch >= self.min_epochs and self.counter >= self.patience


class Trainer:
    def __init__(self, overrides: dict | None = None, cfg: dict | None = None,
                 model: DetectionModel | None = None):
        # Validate after merging saved settings, so partial resume overrides are
        # checked in the checkpoint's context rather than against fresh defaults.
        self.cfg = dict(cfg) if cfg is not None else merge_overrides(load_config(), overrides or {})
        self._explicit_model = model
        self.resume_path: Path | None = None
        self.resume_ckpt = self._locate_resume()
        if self.resume_ckpt is not None:
            from trainer.detection.config.resume import resume_config
            self.cfg = resume_config(self.resume_ckpt, cfg if cfg is not None else (overrides or {}))
        else:
            self.cfg = load_config(self.cfg)
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.distributed = self.world_size > 1
        self.is_main = self.rank == 0
        self._owns_process_group = False
        if self.distributed:
            if not torch.cuda.is_available():
                raise RuntimeError("distributed training requires CUDA and one torchrun process per GPU")
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")
                self._owns_process_group = True
        else:
            self.device = select_device(str(self.cfg["device"]))
        init_seeds(int(self.cfg["seed"]) + self.rank, bool(self.cfg["deterministic"]))
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        self.run_dir = self._setup_run_dir()
        self.metrics_result: dict = {}
        self.best_fitness = -math.inf
        self.global_step = 0
        self.attempted_steps = 0
        self.batch_step = 0
        self.best_epoch = -1
        self.best_model_state = None
        self.selected_checkpoint: Path | None = None
        self.provenance = {}
        self.start_epoch = 0
        self._tracker = None
        self._frozen_modules: list[nn.Module] = []
        self.distiller = None
        self.routing_manifest = []
        self.prefix_training_seconds = 0.

    # -- setup -----------------------------------------------------------------
    def _locate_resume(self) -> dict | None:
        resume = self.cfg.get("resume")
        if not resume:
            return None
        if isinstance(resume, str) and resume not in ("true", "True"):
            path = Path(resume)
        else:
            path = Path(self.cfg["project"]) / self.cfg["name"] / "weights" / "last.pt"
        if not path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {path}")
        self.resume_path = path.resolve()
        LOGGER.info(f"resuming from {path}")
        return load_checkpoint(path)

    def _setup_run_dir(self) -> Path:
        run_dir: Path | None = None
        if self.is_main:
            if self.resume_ckpt is not None and self.cfg["resume_mode"] == "exact":
                if self.resume_path is not None and self.resume_path.parent.name == "weights":
                    run_dir = self.resume_path.parent.parent
                else:
                    run_dir = Path(self.resume_ckpt["run_dir"])
            else:
                from trainer.detection.engine.val_writer import write_cfg
                name = self.cfg["name"]
                if self.resume_ckpt is not None and self.cfg["resume_mode"] == "continue":
                    name = f"{name}_continue"
                run_dir = increment_path(Path(self.cfg["project"]) / name,
                                         exist_ok=bool(self.cfg["exist_ok"]) and self.resume_ckpt is None)
                (run_dir / "weights").mkdir(parents=True, exist_ok=True)
                write_cfg(run_dir, self.cfg, str(self.cfg["data"]), str(self.cfg["model"]))
        if self.distributed:
            payload = [str(run_dir) if run_dir is not None else None]
            dist.broadcast_object_list(payload, src=0)
            run_dir = Path(payload[0])
        assert run_dir is not None
        return run_dir

    def _setup_file_log(self) -> logging.FileHandler | None:
        if not self.is_main:
            return None
        log_path = self.run_dir / "logs" / "train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self.resume_ckpt is not None else "w"
        handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        LOGGER.addHandler(handler)
        return handler

    def _teardown_file_log(self, handler: logging.FileHandler | None) -> None:
        if handler is None:
            return
        handler.flush()
        handler.close()
        LOGGER.removeHandler(handler)

    def _build_model(self, nc: int) -> DetectionModel:
        if self.resume_ckpt is not None:
            checkpoint_nc = int(self.resume_ckpt["nc"])
            if checkpoint_nc != nc:
                raise ValueError(
                    f"resume checkpoint has {checkpoint_nc} classes but dataset defines {nc}; "
                    "use finetune instead of resume"
                )
            return model_from_checkpoint(self.resume_ckpt, nc=nc)
        if self._explicit_model is not None:
            return self._explicit_model
        spec = str(self.cfg["model"])
        if spec.endswith(".pt"):
            ckpt = load_checkpoint(spec)
            model = model_from_checkpoint(ckpt, nc=nc)
            model.load_weights(ckpt["model"])
            LOGGER.info(f"finetuning from {spec}")
            return model
        return DetectionModel(spec, nc=nc, reg_max=int(self.cfg["reg_max"]), reg_channels=int(self.cfg["reg_channels"]))

    def _freeze(self, model: DetectionModel) -> None:
        n = int(self.cfg["freeze"])
        if n <= 0:
            return
        frozen = 0
        for layer in model.layers:
            if layer.i < n:
                for p in layer.parameters():
                    p.requires_grad_(False)
                self._frozen_modules.append(layer)
                frozen += 1
        LOGGER.info(f"froze first {frozen} module(s)")

    def _keep_frozen_modules_eval(self) -> None:
        for module in self._frozen_modules:
            module.eval()

    def _build_optimizer(self, model: nn.Module, iterations: int) -> torch.optim.Optimizer:
        name = str(self.cfg["optimizer"]).lower()
        if name == "auto":
            name = "sgd" if iterations > 10000 else "adamw"
        self.cfg["optimizer"] = name  # persist the effective optimizer, not "auto"
        extra = self.distiller.adapters if self.distiller is not None else None
        biases, no_decay, decay, rest, qkv, names, self.routing_manifest = optimizer_partition(model, self.cfg, extra)
        if name != "musgd":
            decay = decay + rest + qkv
        lr0 = float(self.cfg["lr0"])
        momentum = float(self.cfg["momentum"])
        wd = float(self.cfg["weight_decay"])
        detail = ""
        if name == "sgd":
            opt = torch.optim.SGD([{"params": biases, "lr": lr0},
                                   {"params": no_decay, "weight_decay": 0.0},
                                   {"params": decay, "weight_decay": wd}],
                                  lr=lr0, momentum=momentum, nesterov=True)
        elif name == "adamw":
            lr0 = min(lr0, 0.002)
            opt = torch.optim.AdamW([{"params": biases, "lr": lr0},
                                     {"params": no_decay, "weight_decay": 0.0},
                                     {"params": decay, "weight_decay": wd}],
                                    lr=lr0, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "musgd":
            muon = decay
            groups = [{"params": biases, "lr": lr0},
                      {"params": no_decay, "weight_decay": 0.0},
                      {"params": muon, "weight_decay": wd, "use_muon": True}]
            if rest:
                groups.append({"params": rest, "weight_decay": wd})
            if qkv:
                groups.append({"params": qkv, "weight_decay": wd, "use_muon": True, "muon_splits": 3})
            mw, sw = float(self.cfg["muon_weight"]), float(self.cfg["sgd_weight"])
            opt = MuSGD(groups, lr=lr0, momentum=momentum, nesterov=True,
                        muon_weight=mw, sgd_weight=sw, muon_beta=float(self.cfg["muon_beta"]),
                        ns_steps=int(self.cfg["muon_ns_steps"]), batch_ns=bool(self.cfg["muon_batch_ns"]))
            opt.profile = bool(self.cfg["profile"])
            detail = f", {len(muon)} of the decay params orthogonalized (muon={mw}, sgd={sw})"
        else:
            raise ValueError(f"unknown optimizer '{name}'")
        for group in opt.param_groups:
            group["param_names"] = [names[id(p)] for p in group["params"]]
        self.cfg["lr0"] = lr0
        LOGGER.info(f"optimizer: {name} (lr0={lr0}, {len(decay)} decay / {len(no_decay)} no-decay / "
                    f"{len(biases)} bias params{detail})")
        return opt

    def _lr_lambda(self):
        lrf = float(self.cfg["lrf"])
        epochs = int(self.cfg["epochs"])
        if self.cfg["cos_lr"]:
            return lambda e: lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * e / max(epochs, 1)))
        return lambda e: 1 - (1 - lrf) * e / max(epochs, 1)

    @staticmethod
    def _batch_bar(done: int, total: int, width: int = 25) -> str:
        filled = int(width * done / total)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {done}/{total} batches, {total - done} remaining"

    # -- training ----------------------------------------------------------------
    def train(self) -> dict:
        log_handler = self._setup_file_log()
        try:
            return self._train()
        finally:
            if self._tracker is not None:
                self._tracker.finish()
                self._tracker = None
            self._teardown_file_log(log_handler)
            if self._owns_process_group and dist.is_initialized():
                dist.destroy_process_group()

    def _train(self) -> dict:
        from trainer.detection.engine.val_writer import write_report
        cfg = self.cfg
        resume_run_id = (self.resume_ckpt.get("mlflow_run_id")
                         if self.resume_ckpt is not None and cfg["resume_mode"] == "exact" else None)
        tracker = (build_tracker(cfg, self.run_dir, resume_run_id=resume_run_id)
                   if self.is_main else MultiTracker([]))
        self._tracker = tracker
        data = parse_dataset_toml(cfg["data"])

        model = self._build_model(nc=data["nc"]).to(self.device)
        if model.nc != data["nc"]:
            raise ValueError(f"model has {model.nc} classes but dataset defines {data['nc']}")
        model.names = data["names"]
        from trainer.detection.engine.val_writer import write_cfg

        # autobatch runs before any DataLoaders are created so no /dev/shm pressure
        if cfg.get("autobatch") and self.device.type == "cuda" and self.resume_ckpt is None:
            selected = (autobatch(
                model, int(cfg["imgsz"]), self.device, amp=bool(cfg["amp"]),
                optimizer=str(cfg["optimizer"]), include_ema=True,
            ) if self.is_main else 0)
            if self.distributed:
                value = torch.tensor([selected], device=self.device, dtype=torch.int64)
                dist.broadcast(value, src=0)
                selected = int(value.item())
            cfg["batch"] = selected

        train_loader = build_dataloader(
            data, cfg, "train", rank=self.rank, world_size=self.world_size,
        )
        _val_batch_cfg = int(cfg.get("val_batch", 0))
        val_batch = _val_batch_cfg if _val_batch_cfg > 0 else int(cfg["batch"])
        val_loader = build_dataloader(data, cfg, "val", batch=val_batch) if self.is_main else None
        if cfg.get("channels_last") and self.device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        self._freeze(model)
        if cfg["teacher"]:
            from trainer.detection.engine.distillation import Distiller
            self.distiller = Distiller(model, cfg, data["names"], self.device)
        if cfg.get("compile"):
            model = torch.compile(model)

        epochs = int(cfg["epochs"])
        nb = len(train_loader)
        accumulate = max(round(int(cfg["nbs"]) / (int(cfg["batch"]) * self.world_size)), 1)
        loss_fn = DetectionLoss(ModelEMA._unwrap(model), cfg)
        optimizer = self._build_optimizer(model, iterations=epochs * math.ceil(nb / accumulate))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._lr_lambda())
        base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        amp_enabled = bool(cfg["amp"]) and self.device.type == "cuda"
        scaler = torch.amp.GradScaler(enabled=amp_enabled)
        ema = ModelEMA(model, decay_max=float(cfg["ema_decay"]), warmup_steps=int(cfg["ema_warmup_steps"]))
        if cfg.get("compile"):
            ema.ema = torch.compile(ema.ema)
        if self.distributed:
            model = DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank)
            if self.distiller is not None and self.distiller.adapters is not None:
                self.distiller.adapters = DistributedDataParallel(
                    self.distiller.adapters, device_ids=[self.local_rank], output_device=self.local_rank,
                )
        stopper = EarlyStopper(cfg)
        if self.resume_ckpt is None:
            tracker.log_params(cfg)
        raw_model = ModelEMA._unwrap(model)
        if self.is_main:
            self.provenance = collect_provenance(
                self.run_dir, cfg, raw_model.architecture_spec(),
                {"train": train_loader, "val": val_loader}, self.device,
            )
            if self.distiller is not None:
                self.provenance["distillation"] = {
                    "teacher": str(Path(cfg["teacher"]).resolve()), "teacher_sha256": self.distiller.teacher_sha256,
                    "teacher_parameters": self.distiller.teacher.num_params(),
                    "adapter_parameters": sum(p.numel() for p in self.distiller.adapters.parameters())
                    if self.distiller.adapters is not None else 0,
                    "mask": "GT foreground feature-cell centers on identically augmented training images",
                }
            if self.resume_ckpt is not None:
                self.provenance["parent_checkpoint"] = {
                    "path": str(self.resume_path), "sha256": file_sha256(self.resume_path),
                    "mode": cfg["resume_mode"],
                    "completed_epochs": int(self.resume_ckpt["epoch"]) + 1,
                    "shared_prefix_training_seconds": self.resume_ckpt.get("training_elapsed_seconds"),
                }
        tracker.set_tags({"model": raw_model.model_name, "dataset": str(cfg["data"])})

        if self.resume_ckpt is not None:
            self._restore(model, optimizer, scheduler, scaler, ema, stopper, tracker)
        if self.is_main:
            write_cfg(self.run_dir, cfg, str(cfg["data"]), raw_model.architecture_spec())
            write_provenance(self.run_dir, self.provenance)
            write_json(self.run_dir / "cfg" / "optimizer_routing.json", self.routing_manifest)
            tracker.log_artifact(self.run_dir / "cfg", artifact_path="cfg")

        warmup_steps = (max(round(float(cfg["warmup_epochs"]) * nb), min(100, nb))
                        if float(cfg["warmup_epochs"]) > 0 else 0)
        close_epochs = max(int(cfg["close_mosaic"]), 0)
        close_at = max(epochs - close_epochs, 0)
        strides_max = int(raw_model.stride.max())
        # Scheduler base LRs survive resume; current optimizer LRs do not
        # represent the warmup target after an interrupted run.
        base_lrs = list(scheduler.base_lrs)

        LOGGER.info(raw_model.info())
        LOGGER.info(f"training {epochs} epochs on {len(train_loader.dataset)} images "
                    f"(run dir: {self.run_dir})")
        LOGGER.info(HEADER)
        t_start = time.time()
        self._training_timer = time.perf_counter()

        use_channels_last = bool(cfg.get("channels_last")) and self.device.type == "cuda"
        use_multiscale = bool(cfg.get("multiscale"))
        multiscale_prob = float(cfg["multiscale_prob"]) if use_multiscale else 0.0
        multiscale_range = cfg["multiscale_range"] if use_multiscale else None
        warmup_bias_lr = float(cfg["warmup_bias_lr"])
        warmup_momentum = float(cfg["warmup_momentum"])
        base_momentum = float(cfg["momentum"])
        profiler = StepProfiler(cfg["profile"], self.device)
        first_training_batch = True

        for epoch in range(self.start_epoch, epochs):
            profiler.reset()
            if cfg["profile"] and self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(cuda_device_index(self.device))
            loss_diagnostics = {}
            kd_epoch = torch.zeros(3, device=self.device)
            if isinstance(optimizer, MuSGD):
                for group in optimizer.param_groups:
                    group["muon_weight"] = muon_weight_at_epoch(cfg, epoch)
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            if close_epochs and epoch >= close_at and (epoch == close_at or epoch == self.start_epoch):
                LOGGER.info(f"epoch {epoch + 1}: mosaic/mixup/copypaste disabled (close_mosaic)")
            model.train()
            self._keep_frozen_modules_eval()
            epoch_loss = torch.zeros(3, device=self.device)
            t_epoch = time.time()
            data_wait = 0.0
            skipped_steps = clipped_steps = attempted_steps = 0
            optimizer.zero_grad(set_to_none=True)
            t_fetch = time.time()

            for i, batch in enumerate(train_loader):
                data_wait += time.time() - t_fetch
                step = epoch * nb + i
                self.batch_step = step + 1
                is_sync = (i + 1) % accumulate == 0 or (i + 1) == nb
                loss_divisor = accumulation_divisor(i, nb, accumulate)

                # per-step LR + momentum warmup (keyed on global step)
                if warmup_steps and step <= warmup_steps:
                    t = step / warmup_steps
                    lr_scale = self._lr_lambda()(epoch)
                    for j, pg in enumerate(optimizer.param_groups):
                        start_lr = warmup_bias_lr if j == 0 else 0.0
                        pg["lr"] = start_lr + (base_lrs[j] * lr_scale - start_lr) * t
                        if "momentum" in pg:
                            pg["momentum"] = warmup_momentum + (base_momentum - warmup_momentum) * t

                imgs = batch["img"].to(self.device, non_blocking=True).float() / 255.0
                if use_channels_last:
                    imgs = imgs.to(memory_format=torch.channels_last)
                if use_multiscale and random.random() < multiscale_prob:
                    lo, hi = multiscale_range
                    target = random.uniform(lo, hi) * int(cfg["imgsz"])
                    target = max(int(round(target / strides_max) * strides_max), strides_max)
                    if target != imgs.shape[-1]:
                        imgs = F.interpolate(imgs, size=(target, target), mode="bilinear",
                                             align_corners=False)

                with ExitStack() as sync_stack:
                    sync_stack.enter_context(nullcontext() if is_sync else getattr(model, "no_sync", nullcontext)())
                    if self.distiller is not None and self.distiller.adapters is not None and not is_sync:
                        sync_stack.enter_context(getattr(self.distiller.adapters, "no_sync", nullcontext)())
                    with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                        timing = "first_forward_loss" if first_training_batch else "forward_loss"
                        with profiler.measure(timing):
                            if self.distiller is not None:
                                feats, features = model(imgs, return_features=True)
                            else:
                                feats = model(imgs)
                            loss, components = loss_fn(feats, batch)
                        if self.distiller is not None:
                            with profiler.measure("teacher_forward"):
                                teacher_output = self.distiller.teacher_forward(imgs)
                            with profiler.measure("distillation_loss"):
                                kd_loss, kd_terms = self.distiller.loss(feats, features, teacher_output, batch)
                                loss = loss + kd_loss
                                kd_epoch += torch.stack([kd_terms[key] for key in ("feature", "cls", "box")])
                    with profiler.measure("backward"):
                        scaler.scale(loss / loss_divisor).backward()
                first_training_batch = False
                if cfg["loss_diagnostics"]:
                    for key, value in loss_fn.last_diagnostics.items():
                        loss_diagnostics[key] = loss_diagnostics.get(key, 0) + value
                if is_sync:
                    if isinstance(optimizer, MuSGD):
                        interval = int(cfg["muon_log_interval"])
                        optimizer.collect_diagnostics = bool(interval and self.attempted_steps % interval == 0)
                        optimizer.last_diagnostics = []
                        optimizer.ns_seconds = 0.
                    with profiler.measure("optimizer_scaler_clip_ema"):
                        applied, grad_norm = (apply_optimizer_step(scaler, optimizer, model, ema, profiler=profiler)
                                              if cfg["profile"] else apply_optimizer_step(scaler, optimizer, model, ema))
                    if isinstance(optimizer, MuSGD):
                        if cfg["profile"]:
                            profiler.values["newton_schulz_subset"] = profiler.values.get("newton_schulz_subset", 0.) + optimizer.ns_seconds
                        if applied and optimizer.last_diagnostics and self.is_main:
                            append_jsonl(self.run_dir / "logs" / "muon_updates.jsonl", {
                                "epoch": epoch + 1, "attempted_step": self.attempted_steps + 1,
                                "successful_step": self.global_step + 1, "layers": optimizer.last_diagnostics,
                            })
                    self.attempted_steps += 1
                    attempted_steps += 1
                    skipped_steps += int(not applied)
                    clipped_steps += int(math.isfinite(grad_norm) and grad_norm > 10.0)
                    self.global_step += int(applied)
                epoch_loss += components
                if self.is_main and (i % 5 == 0 or (i + 1) == nb):
                    print(f"\r  epoch {epoch + 1}/{epochs}  {self._batch_bar(i + 1, nb)}",
                          end="", flush=True)
                profiler.sync()
                t_fetch = time.time()
            if self.is_main:
                print()  # finalize the progress bar line
            lr_now = optimizer.param_groups[-1]["lr"]
            scheduler.step()
            if self.distributed:
                dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
                epoch_loss /= self.world_size
            # Pre-start val workers before the GPU sync so disk reads overlap.
            should_validate = (epoch + 1) % int(cfg["val_period"]) == 0 or epoch == epochs - 1
            val_iter = iter(val_loader) if (self.is_main and should_validate) else None
            mean_loss = (epoch_loss / nb).tolist()
            epoch_time = time.time() - t_epoch
            data_pct = 100.0 * data_wait / max(epoch_time, 1e-9)
            if data_pct > 5.0:
                LOGGER.warning(f"dataloader is a bottleneck: {data_pct:.0f}% of epoch spent waiting for data")

            metrics_row = {"train/box": mean_loss[0], "train/cls": mean_loss[1], "train/dfl": mean_loss[2],
                           "lr": lr_now, "epoch_time": epoch_time,
                           "data_wait": data_wait, "data_pct": data_pct}
            metrics_row.update({"optimizer/attempted": attempted_steps, "optimizer/skipped": skipped_steps,
                                "optimizer/clipped": clipped_steps,
                                "optimizer/successful_total": self.global_step})
            if isinstance(optimizer, MuSGD):
                metrics_row["optimizer/muon_weight"] = muon_weight_at_epoch(cfg, epoch)
            if self.distiller is not None:
                if self.distributed:
                    dist.all_reduce(kd_epoch)
                    kd_epoch /= self.world_size
                metrics_row.update({f"train/kd_{key}": float(value / nb)
                                    for key, value in zip(("feature", "cls", "box"), kd_epoch)})
            if cfg["loss_diagnostics"]:
                if self.distributed:
                    # All ranks must reduce the same keys even if some have no positives.
                    keys = ("gt_count", "gts_with_positives", "positive_anchors", "clipped_dfl_sides", "total_dfl_sides",
                            "positive_small", "positive_medium", "positive_large", "assignment_overlap_sum",
                            "overlap_small_sum", "overlap_medium_sum", "overlap_large_sum")
                    for key in keys:
                        value = torch.as_tensor(loss_diagnostics.get(key, 0.), device=self.device, dtype=torch.float64)
                        dist.all_reduce(value)
                        loss_diagnostics[key] = value
                diag = {key: float(value) for key, value in loss_diagnostics.items()}
                diag["clipped_dfl_fraction"] = diag.get("clipped_dfl_sides", 0) / max(diag.get("total_dfl_sides", 0), 1)
                diag["positives_per_gt"] = diag.get("positive_anchors", 0) / max(diag.get("gt_count", 0), 1)
                diag["mean_assignment_overlap"] = diag.get("assignment_overlap_sum", 0) / max(diag.get("positive_anchors", 0), 1)
                for size in ("small", "medium", "large"):
                    diag[f"mean_overlap_{size}"] = diag.get(f"overlap_{size}_sum", 0) / max(diag.get(f"positive_{size}", 0), 1)
                metrics_row.update({f"diagnostics/{key}": value for key, value in diag.items()})
            result = None
            if should_validate:
                if self.is_main:
                    assert val_loader is not None
                    with profiler.measure("validation"):
                        result = validate(ema.ema, val_loader, cfg, data["names"], device=self.device,
                                         half=amp_enabled, compute_loss=bool(cfg["val_compute_loss"]),
                                         prefetched_iter=val_iter)
                if self.distributed:
                    payload = [result]
                    dist.broadcast_object_list(payload, src=0, device=self.device)
                    result = payload[0]
                self.metrics_result = result
                metrics_row.update({"val/mAP50": result["mAP50"], "val/mAP50-95": result["mAP50-95"],
                                    "val/precision": result["precision"], "val/recall": result["recall"]})
                if result["val_loss"] is not None:
                    metrics_row["val/loss"] = result["val_loss"]
                if cfg.get("log_per_class"):
                    metrics_row.update(per_class_metrics(result, data["names"]))
                metrics_row.update(comparison_metrics(result))
            if cfg["profile"] and self.is_main:
                profile_values = {**profiler.values, "data_wait": data_wait}
                if self.device.type == "cuda":
                    profile_values["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
                        cuda_device_index(self.device)
                    )
                append_jsonl(self.run_dir / "logs" / "profile.jsonl", {"epoch": epoch + 1, "rank": self.rank, **profile_values})
                metrics_row.update({f"profile/{key}": value for key, value in profile_values.items()})
            tracker.log_metrics(metrics_row, step=epoch)

            r = result or {}
            LOGGER.info(f"{epoch + 1:>4d}/{epochs:<4d} {mean_loss[0]:>8.3f} {mean_loss[1]:>8.3f} "
                        f"{mean_loss[2]:>8.3f} {r.get('mAP50', 0.0):>7.3f} {r.get('mAP50-95', 0.0):>9.3f} "
                        f"{r.get('precision', 0.0):>6.3f} {r.get('recall', 0.0):>6.3f} {lr_now:>9.5f} "
                        f"{data_pct:>5.1f}% {format_seconds(epoch_time):>8s}")

            fitness = self._fitness(result, mean_loss)
            improved = fitness > self.best_fitness
            if improved:
                self.best_fitness = fitness
                self.best_epoch = epoch
                if self.is_main:
                    self.best_model_state = {
                        key: value.detach().cpu().clone()
                        for key, value in ModelEMA._unwrap(ema.ema).state_dict().items()
                    }
            should_stop = result is not None and stopper.step(fitness, epoch)
            self._save_checkpoint(model, ema, optimizer, scheduler, scaler, stopper, tracker, epoch,
                                  data["names"], best=improved)
            if should_stop:
                LOGGER.info(f"early stopping at epoch {epoch + 1} "
                            f"(best epoch {stopper.best_epoch + 1}, best {cfg['early_stop_metric']}="
                            f"{stopper.best:.4f})")
                break

        total_time = time.time() - t_start
        best_pt = self.run_dir / "weights" / "best.pt"
        LOGGER.info(f"Best: {cfg['early_stop_metric']}={self.best_fitness:.4f} | "
                    f"Saved: {best_pt} | Total: {format_seconds(total_time)}")
        if self.is_main:
            assert val_loader is not None
            restore_best = bool(cfg["early_stop_restore_best"]) and self.best_model_state is not None
            selected_state = self.best_model_state if restore_best else ModelEMA._unwrap(ema.ema).state_dict()
            raw_model.load_state_dict(selected_state)
            selected_epoch = self.best_epoch if restore_best else int(scheduler.last_epoch) - 1
            # Standalone inference artifact: never attach last-epoch optimizer state to best weights.
            self.selected_checkpoint = self.run_dir / "weights" / "selected.pt"
            atomic_torch_save({
                "checkpoint_version": CHECKPOINT_VERSION,
                "model": {k: v.detach().cpu() for k, v in raw_model.state_dict().items()},
                "epoch": selected_epoch, "cfg": checkpoint_safe(cfg), "names": data["names"],
                "model_name": raw_model.model_name, "model_cfg": raw_model.architecture_spec(),
                "nc": raw_model.nc, "ch": raw_model.ch, "reg_max": raw_model.head.reg_max,
                "provenance": self.provenance,
            }, self.selected_checkpoint)
            self.metrics_result = validate(raw_model, val_loader, cfg, data["names"], device=self.device,
                                           half=amp_enabled, compute_loss=bool(cfg["val_compute_loss"]))
            self.metrics_result["checkpoint"] = {
                "path": str(self.selected_checkpoint), "sha256": file_sha256(self.selected_checkpoint),
                "epoch": selected_epoch + 1, "selection": "best EMA" if restore_best else "last EMA",
            }
            from trainer.detection.engine.val_writer import write_logs
            write_logs(self.run_dir, self.metrics_result, data["names"])
            write_report(self.run_dir, self.metrics_result, data["names"], cfg)
            LOGGER.info(per_class_table(self.metrics_result, data["names"]))
            artifacts = [p for folder in ("cfg", "logs", "weights", "plots", "reports")
                         for p in (self.run_dir / folder).rglob("*") if p.is_file()]
            write_json(self.run_dir / "manifest.json", {
                "selected_checkpoint": self.metrics_result["checkpoint"],
                "files": {str(p.relative_to(self.run_dir)): file_sha256(p) for p in artifacts},
            })
            for folder in ("cfg", "logs", "weights", "plots", "reports"):
                if (self.run_dir / folder).exists():
                    tracker.log_artifact(self.run_dir / folder, artifact_path=folder)
            tracker.log_artifact(self.run_dir / "manifest.json")
        if self.distributed:
            payload = [self.metrics_result]
            dist.broadcast_object_list(payload, src=0, device=self.device)
            self.metrics_result = payload[0]
        return self.metrics_result

    # -- state -----------------------------------------------------------------
    def _fitness(self, result: dict | None, mean_loss: list[float]) -> float:
        metric = str(self.cfg["early_stop_metric"])
        if result is None:
            return self.best_fitness  # no val this epoch: keep best
        if metric == "val_loss":
            return -float(result["val_loss"])
        return float(result.get(metric, 0.0))

    def _save_checkpoint(self, model, ema, optimizer, scheduler, scaler, stopper, tracker, epoch,
                         names, best: bool) -> None:
        rng_by_rank = [_rng_state()]
        if self.distributed:
            rng_by_rank = [None] * self.world_size
            dist.all_gather_object(rng_by_rank, _rng_state())
        if not self.is_main:
            return
        raw = ModelEMA._unwrap(model)
        ckpt = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "epoch": epoch,
            "global_step": self.global_step,
            "attempted_steps": self.attempted_steps,
            "batch_step": self.batch_step,
            "data_rng_version": DATA_RNG_VERSION,
            "world_size": self.world_size,
            "provenance": self.provenance,
            "model": ModelEMA._unwrap(ema.ema).state_dict(),
            "raw_model": ModelEMA._unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "ema_meta": {"step_count": ema.step_count, "decay_max": ema.decay_max,
                         "warmup_steps": ema.warmup_steps},
            "stopper": {"best": stopper.best, "best_epoch": stopper.best_epoch,
                        "counter": stopper.counter},
            "best_fitness": self.best_fitness,
            "best_epoch": self.best_epoch,
            "best_model": self.best_model_state,
            "cfg": checkpoint_safe(self.cfg),
            "names": checkpoint_safe(names),
            "model_name": raw.model_name,
            "model_cfg": checkpoint_safe(raw.architecture_spec()),
            "model_scale": raw.scale,
            "nc": raw.nc,
            "ch": raw.ch,
            "reg_max": raw.head.reg_max,
            "run_dir": str(self.run_dir),
            "mlflow_run_id": tracker.run_id,
            "rng_state": _rng_state(),
            "rng_by_rank": rng_by_rank,
            "distillation": self.distiller.state_dict() if self.distiller is not None else None,
            "training_elapsed_seconds": (None if self.prefix_training_seconds is None else
                                          self.prefix_training_seconds + time.perf_counter() - self._training_timer),
        }
        weights_dir = self.run_dir / "weights"
        atomic_torch_save(ckpt, weights_dir / "last.pt")
        if best:
            atomic_torch_save(ckpt, weights_dir / "best.pt")
        period = int(self.cfg["save_period"])
        if period > 0 and (epoch + 1) % period == 0:
            atomic_torch_save(ckpt, weights_dir / f"epoch_{epoch + 1}.pt")

    def _restore(self, model, optimizer, scheduler, scaler, ema, stopper, tracker) -> None:
        from trainer.detection.api import _strip_compiled_prefix
        ckpt = self.resume_ckpt
        self.prefix_training_seconds = ckpt.get("training_elapsed_seconds")
        ModelEMA._unwrap(model).load_state_dict(_strip_compiled_prefix(ckpt["raw_model"]))
        ModelEMA._unwrap(ema.ema).load_state_dict(_strip_compiled_prefix(ckpt["model"]))
        ema.step_count = ckpt["ema_meta"]["step_count"]
        ema.decay_max = ckpt["ema_meta"]["decay_max"]
        ema.warmup_steps = ckpt["ema_meta"]["warmup_steps"]
        exact = self.cfg["resume_mode"] in ("exact", "branch")
        if self.distiller is not None:
            self.distiller.load_state_dict(ckpt.get("distillation"))
        elif ckpt.get("distillation") is not None:
            raise ValueError("cannot resume distillation without its teacher and adapter state")
        if exact and int(ckpt.get("world_size", 1)) != self.world_size:
            raise ValueError("exact resume requires the original world_size")
        replay_error = None
        if exact and self.is_main:
            old_data = ckpt.get("provenance", {}).get("datasets")
            if old_data and old_data != self.provenance.get("datasets"):
                replay_error = "exact resume dataset identity changed"
            old_code = ckpt.get("provenance", {}).get("training_code_sha256")
            if old_code and old_code != self.provenance.get("training_code_sha256"):
                replay_error = "exact resume training code changed; use resume_mode='continue'"
        if self.distributed:
            payload = [replay_error]
            dist.broadcast_object_list(payload, src=0, device=self.device)
            replay_error = payload[0]
        if replay_error:
            raise ValueError(replay_error)
        group = ckpt["optimizer"]["param_groups"][0]
        saved_optimizer = "adamw" if "betas" in group else ("musgd" if "muon_weight" in group else "sgd")
        if saved_optimizer != self.cfg["optimizer"]:
            raise ValueError(f"checkpoint optimizer {saved_optimizer} != requested {self.cfg['optimizer']}")
        continuation_lrs = list(scheduler.base_lrs)
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        else:  # v1 checkpoints predate scheduler persistence
            scheduler.last_epoch = int(ckpt["epoch"]) + 1
            scheduler._step_count = scheduler.last_epoch + 1
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        if not exact:
            scheduler.base_lrs = continuation_lrs
            resumed_epoch = int(ckpt["epoch"]) + 1
            for group, base in zip(optimizer.param_groups, continuation_lrs):
                group["initial_lr"] = base
                group["lr"] = base * self._lr_lambda()(resumed_epoch)
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        scaler.load_state_dict(ckpt["scaler"])
        if exact:
            stopper.best = ckpt["stopper"]["best"]
            stopper.best_epoch = ckpt["stopper"]["best_epoch"]
            stopper.counter = ckpt["stopper"]["counter"]
            self.best_fitness = ckpt["best_fitness"]
            self.best_epoch = ckpt["best_epoch"]
            self.best_model_state = ckpt["best_model"]
        self.global_step = ckpt["global_step"]
        self.attempted_steps = ckpt.get("attempted_steps", self.global_step)
        self.batch_step = ckpt.get("batch_step", 0)
        self.start_epoch = ckpt["epoch"] + 1
        states = ckpt.get("rng_by_rank", [])
        _restore_rng_state(states[self.rank] if self.rank < len(states) else ckpt.get("rng_state"))
        if self.cfg["resume_mode"] == "exact":
            tracker.resume(ckpt.get("mlflow_run_id"), start_step=self.start_epoch)
        LOGGER.info(f"resumed at epoch {self.start_epoch}")
