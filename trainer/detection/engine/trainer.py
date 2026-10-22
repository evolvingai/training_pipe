"""Single-GPU training loop: AMP, EMA, warmup + cosine/linear LR, gradient
accumulation, resume, early stopping, checkpointing, tracker logging."""

from __future__ import annotations

import logging
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.detection.config import load_config
from trainer.detection.data import build_dataloader, parse_dataset_toml
from trainer.detection.engine.validator import log_per_class_metrics, per_class_table, validate
from trainer.detection.nn.loss import DetectionLoss
from trainer.detection.nn.model import DetectionModel
from trainer.detection.trackers import build_tracker
from trainer.detection.utils.checks import autobatch, init_seeds, select_device
from trainer.detection.utils.ema import ModelEMA
from trainer.detection.utils.logging import LOGGER, format_seconds, increment_path

HEADER = (f"{'Epoch':>9s} {'box':>8s} {'cls':>8s} {'dfl':>8s} {'mAP50':>7s} {'mAP50-95':>9s} "
          f"{'P':>6s} {'R':>6s} {'LR':>9s} {'data%':>6s} {'time':>8s}")


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
        self.cfg = cfg or load_config(overrides)
        self._explicit_model = model
        self.device = select_device(str(self.cfg["device"]))
        init_seeds(int(self.cfg["seed"]), bool(self.cfg["deterministic"]))
        self.resume_ckpt = self._locate_resume()
        self.run_dir = self._setup_run_dir()
        self.metrics_result: dict = {}
        self.best_fitness = -math.inf
        self.global_step = 0
        self.start_epoch = 0

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
        LOGGER.info(f"resuming from {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def _setup_run_dir(self) -> Path:
        if self.resume_ckpt is not None:
            return Path(self.resume_ckpt["run_dir"])
        from trainer.detection.engine.val_writer import write_cfg
        run_dir = increment_path(Path(self.cfg["project"]) / self.cfg["name"],
                                 exist_ok=bool(self.cfg["exist_ok"]))
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        write_cfg(run_dir, self.cfg, str(self.cfg["data"]), str(self.cfg["model"]))
        return run_dir

    def _setup_file_log(self) -> logging.FileHandler:
        log_path = self.run_dir / "logs" / "train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self.resume_ckpt is not None else "w"
        handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        LOGGER.addHandler(handler)
        return handler

    def _teardown_file_log(self, handler: logging.FileHandler) -> None:
        handler.flush()
        handler.close()
        LOGGER.removeHandler(handler)

    def _build_model(self, nc: int) -> DetectionModel:
        if self._explicit_model is not None:
            return self._explicit_model
        spec = str(self.cfg["model"])
        if spec.endswith(".pt"):
            ckpt = torch.load(spec, map_location="cpu", weights_only=False)
            model = DetectionModel(ckpt["model_name"], nc=nc)
            model.load_weights(ckpt["model"])
            LOGGER.info(f"finetuning from {spec}")
            return model
        return DetectionModel(spec, nc=nc)

    def _freeze(self, model: DetectionModel) -> None:
        n = int(self.cfg["freeze"])
        if n <= 0:
            return
        frozen = 0
        for layer in model.layers:
            if layer.i < n:
                for p in layer.parameters():
                    p.requires_grad_(False)
                frozen += 1
        LOGGER.info(f"froze first {frozen} module(s)")

    def _build_optimizer(self, model: nn.Module, iterations: int) -> torch.optim.Optimizer:
        decay, no_decay, biases = [], [], []
        for module in model.modules():
            for name, p in module.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                if name == "bias":
                    biases.append(p)
                elif isinstance(module, (nn.BatchNorm2d,)) or p.ndim == 1:
                    no_decay.append(p)
                else:
                    decay.append(p)
        name = str(self.cfg["optimizer"]).lower()
        if name == "auto":
            name = "sgd" if iterations > 10000 else "adamw"
        lr0 = float(self.cfg["lr0"])
        momentum = float(self.cfg["momentum"])
        wd = float(self.cfg["weight_decay"])
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
        else:
            raise ValueError(f"unknown optimizer '{name}'")
        LOGGER.info(f"optimizer: {name} (lr0={lr0}, {len(decay)} decay / {len(no_decay)} no-decay / "
                    f"{len(biases)} bias params)")
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
            self._teardown_file_log(log_handler)

    def _train(self) -> dict:
        cfg = self.cfg
        tracker = build_tracker(cfg, self.run_dir)
        tracker.log_artifact(self.run_dir / "cfg", artifact_path="cfg")
        data = parse_dataset_toml(cfg["data"])
        train_loader = build_dataloader(data, cfg, "train")
        val_loader = build_dataloader(data, cfg, "val", batch=max(int(cfg["batch"]) // 2, 1))

        model = self._build_model(nc=data["nc"]).to(self.device)
        model.names = data["names"]
        if cfg.get("autobatch") and self.device.type == "cuda":
            cfg["batch"] = autobatch(model, int(cfg["imgsz"]), self.device, amp=bool(cfg["amp"]))
            train_loader = build_dataloader(data, cfg, "train")
        if cfg.get("channels_last") and self.device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        self._freeze(model)
        if cfg.get("compile"):
            model = torch.compile(model)

        epochs = int(cfg["epochs"])
        nb = len(train_loader)
        accumulate = max(round(int(cfg["nbs"]) / int(cfg["batch"])), 1)
        loss_fn = DetectionLoss(model, cfg)
        optimizer = self._build_optimizer(model, iterations=epochs * nb)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._lr_lambda())
        amp_enabled = bool(cfg["amp"]) and self.device.type == "cuda"
        scaler = torch.amp.GradScaler(enabled=amp_enabled)
        ema = ModelEMA(model, decay_max=float(cfg["ema_decay"]), warmup_steps=int(cfg["ema_warmup_steps"]))
        stopper = EarlyStopper(cfg)
        tracker.log_params(cfg)
        tracker.set_tags({"model": model.model_name, "dataset": str(cfg["data"])})

        if self.resume_ckpt is not None:
            self._restore(model, optimizer, scaler, ema, stopper, tracker)

        warmup_steps = max(round(float(cfg["warmup_epochs"]) * nb), 100)
        close_at = epochs - int(cfg["close_mosaic"])
        strides_max = int(model.stride.max())
        base_lrs = [pg["lr"] for pg in optimizer.param_groups]

        LOGGER.info(model.info())
        LOGGER.info(f"training {epochs} epochs on {len(train_loader.dataset)} images "
                    f"(run dir: {self.run_dir})")
        LOGGER.info(HEADER)
        t_start = time.time()

        for epoch in range(self.start_epoch, epochs):
            if epoch == close_at and close_at > 0:
                train_loader.dataset.transforms.disable({"Mosaic", "MixUp", "CopyPaste"})
                LOGGER.info(f"epoch {epoch + 1}: mosaic/mixup/copypaste disabled (close_mosaic)")
            model.train()
            epoch_loss = torch.zeros(3, device=self.device)
            t_epoch = time.time()
            data_wait = 0.0
            optimizer.zero_grad(set_to_none=True)
            t_fetch = time.time()

            for i, batch in enumerate(train_loader):
                data_wait += time.time() - t_fetch
                step = epoch * nb + i
                is_sync = (i + 1) % accumulate == 0 or (i + 1) == nb

                # per-step LR + momentum warmup (keyed on global step)
                if step < warmup_steps:
                    t = step / warmup_steps
                    for j, pg in enumerate(optimizer.param_groups):
                        start_lr = float(cfg["warmup_bias_lr"]) if j == 0 else 0.0
                        pg["lr"] = start_lr + (base_lrs[j] * self._lr_lambda()(epoch) - start_lr) * t
                        if "momentum" in pg:
                            pg["momentum"] = float(cfg["warmup_momentum"]) + \
                                (float(cfg["momentum"]) - float(cfg["warmup_momentum"])) * t

                imgs = batch["img"].to(self.device, non_blocking=True).float() / 255.0
                if cfg.get("channels_last") and self.device.type == "cuda":
                    imgs = imgs.to(memory_format=torch.channels_last)
                if cfg.get("multiscale") and random.random() < float(cfg["multiscale_prob"]):
                    lo, hi = cfg["multiscale_range"]
                    target = random.uniform(lo, hi) * int(cfg["imgsz"])
                    target = max(int(round(target / strides_max) * strides_max), strides_max)
                    if target != imgs.shape[-1]:
                        imgs = F.interpolate(imgs, size=(target, target), mode="bilinear",
                                             align_corners=False)

                ctx = nullcontext() if is_sync else getattr(model, "no_sync", nullcontext)()
                with ctx:
                    with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                        feats = model(imgs)
                        loss, components = loss_fn(feats, batch)
                    scaler.scale(loss / accumulate).backward()
                if is_sync:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)
                    self.global_step += 1
                epoch_loss += components
                if (i + 1) % max(1, nb // 10) == 0 or (i + 1) == nb:
                    LOGGER.info(f"  epoch {epoch + 1}/{epochs}  {self._batch_bar(i + 1, nb)}")
                t_fetch = time.time()
            scheduler.step()
            mean_loss = (epoch_loss / nb).tolist()
            lr_now = optimizer.param_groups[-1]["lr"]
            epoch_time = time.time() - t_epoch
            data_pct = 100.0 * data_wait / max(epoch_time, 1e-9)
            if data_pct > 5.0:
                LOGGER.warning(f"dataloader is a bottleneck: {data_pct:.0f}% of epoch spent waiting for data")

            metrics_row = {"train/box": mean_loss[0], "train/cls": mean_loss[1], "train/dfl": mean_loss[2],
                           "lr": lr_now, "epoch_time": epoch_time,
                           "data_wait": data_wait, "data_pct": data_pct}
            result = None
            if (epoch + 1) % int(cfg["val_period"]) == 0 or epoch == epochs - 1:
                result = validate(ema.ema, val_loader, cfg, data["names"], device=self.device)
                self.metrics_result = result
                metrics_row.update({"val/mAP50": result["mAP50"], "val/mAP50-95": result["mAP50-95"],
                                    "val/precision": result["precision"], "val/recall": result["recall"]})
                if cfg.get("log_per_class"):
                    log_per_class_metrics(tracker, result, data["names"], step=epoch)
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
            self._save_checkpoint(model, ema, optimizer, scaler, stopper, tracker, epoch,
                                  data["names"], best=improved)
            if result is not None and stopper.step(fitness, epoch):
                LOGGER.info(f"early stopping at epoch {epoch + 1} "
                            f"(best epoch {stopper.best_epoch + 1}, best {cfg['early_stop_metric']}="
                            f"{stopper.best:.4f})")
                break

        total_time = time.time() - t_start
        if self.metrics_result:
            LOGGER.info(per_class_table(self.metrics_result, data["names"]))
        best_pt = self.run_dir / "weights" / "best.pt"
        LOGGER.info(f"Best: {cfg['early_stop_metric']}={self.best_fitness:.4f} | "
                    f"Saved: {best_pt} | Total: {format_seconds(total_time)}")
        if cfg.get("early_stop_restore_best") and best_pt.exists():
            best_ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
            model.load_state_dict(best_ckpt["model"])
        self._write_training_report(data, model, train_loader, val_loader, total_time)
        tracker.log_artifact(self.run_dir / "logs", artifact_path="logs")
        tracker.log_artifact(self.run_dir / "weights", artifact_path="weights")
        tracker.finish()
        return self.metrics_result

    # -- state -----------------------------------------------------------------
    def _fitness(self, result: dict | None, mean_loss: list[float]) -> float:
        metric = str(self.cfg["early_stop_metric"])
        if result is None:
            return self.best_fitness  # no val this epoch: keep best
        if metric == "val_loss":
            return -sum(mean_loss)
        return float(result.get(metric, 0.0))

    def _save_checkpoint(self, model, ema, optimizer, scaler, stopper, tracker, epoch,
                         names, best: bool) -> None:
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": ema.ema.state_dict(),
            "raw_model": ModelEMA._unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "ema_meta": {"step_count": ema.step_count, "decay_max": ema.decay_max,
                         "warmup_steps": ema.warmup_steps},
            "stopper": {"best": stopper.best, "best_epoch": stopper.best_epoch,
                        "counter": stopper.counter},
            "best_fitness": self.best_fitness,
            "cfg": self.cfg,
            "names": names,
            "model_name": ModelEMA._unwrap(model).model_name,
            "nc": ModelEMA._unwrap(model).nc,
            "run_dir": str(self.run_dir),
            "mlflow_run_id": tracker.run_id,
        }
        weights_dir = self.run_dir / "weights"
        torch.save(ckpt, weights_dir / "last.pt")
        if best:
            torch.save(ckpt, weights_dir / "best.pt")
        period = int(self.cfg["save_period"])
        if period > 0 and (epoch + 1) % period == 0:
            torch.save(ckpt, weights_dir / f"epoch_{epoch + 1}.pt")

    def _write_training_report(self, data: dict, model, train_loader, val_loader,
                               total_time: float) -> None:
        from datetime import datetime
        from trainer.detection.utils.logging import format_seconds

        cfg = self.cfg
        result = self.metrics_result
        names = data["names"]
        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        raw = ModelEMA._unwrap(model)
        n_params = sum(p.numel() for p in raw.parameters())

        lines = [
            f"# Training Report",
            f"",
            f"**Experiment**: {self.run_dir.name}  ",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
            f"**Total time**: {format_seconds(total_time)}  ",
            f"",
            f"## Model",
            f"",
            f"| Key | Value |",
            f"|-----|-------|",
            f"| Architecture | {raw.model_name} |",
            f"| Parameters | {n_params:,} |",
            f"| Input size | {cfg.get('imgsz', '?')} |",
            f"| Classes | {data['nc']} |",
            f"",
            f"## Dataset",
            f"",
            f"| Split | Images |",
            f"|-------|--------|",
            f"| train | {n_train:,} |",
            f"| val | {n_val:,} |",
            f"",
            f"**Classes**: {', '.join(names[c] for c in sorted(names))}",
            f"",
            f"## Training Config",
            f"",
            f"| Key | Value |",
            f"|-----|-------|",
        ]
        for key in ("epochs", "batch", "imgsz", "optimizer", "lr0", "lrf", "momentum",
                    "weight_decay", "amp", "cos_lr", "warmup_epochs"):
            if key in cfg:
                lines.append(f"| {key} | {cfg[key]} |")
        lines += [
            f"",
            f"## Final Metrics",
            f"",
        ]
        if result:
            seen = result["classes_seen"]
            lines.append(f"| Class | P | R | AP50 | AP50-95 |")
            lines.append(f"|-------|---|---|------|---------|")
            for c, name in names.items():
                if seen[c]:
                    lines.append(
                        f"| {name} "
                        f"| {result['precision_per_class'][c]:.3f} "
                        f"| {result['recall_per_class'][c]:.3f} "
                        f"| {result['ap50_per_class'][c]:.3f} "
                        f"| {result['ap_per_class'][c]:.3f} |"
                    )
            lines.append(
                f"| **all** "
                f"| {result['precision']:.3f} "
                f"| {result['recall']:.3f} "
                f"| {result['mAP50']:.3f} "
                f"| {result['mAP50-95']:.3f} |"
            )
        else:
            lines.append("_No validation metrics recorded._")

        report_path = self.run_dir / "reports" / "report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        LOGGER.info(f"training report → {report_path}")

    def _restore(self, model, optimizer, scaler, ema, stopper, tracker) -> None:
        ckpt = self.resume_ckpt
        model.load_state_dict(ckpt["raw_model"])
        ema.ema.load_state_dict(ckpt["model"])
        ema.step_count = ckpt["ema_meta"]["step_count"]
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        stopper.best = ckpt["stopper"]["best"]
        stopper.best_epoch = ckpt["stopper"]["best_epoch"]
        stopper.counter = 0  # reset patience on resume
        self.best_fitness = ckpt["best_fitness"]
        self.global_step = ckpt["global_step"]
        self.start_epoch = ckpt["epoch"] + 1
        tracker.resume(ckpt.get("mlflow_run_id"), start_step=self.start_epoch)
        LOGGER.info(f"resumed at epoch {self.start_epoch}")
