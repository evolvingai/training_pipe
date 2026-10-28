"""Model facade: the single entry point for train / val / predict / export / tune."""

from __future__ import annotations

from pathlib import Path

import torch

from trainer.detection.config import load_config, load_toml
from trainer.detection.nn.model import DetectionModel
from trainer.detection.utils.checks import select_device
from trainer.detection.utils.checkpoints import load_checkpoint, model_from_checkpoint
from trainer.detection.utils.logging import LOGGER


def _strip_compiled_prefix(sd: dict) -> dict:
    """Remove _orig_mod. prefix that torch.compile can leave in a state dict."""
    if sd and any(k.startswith("_orig_mod.") for k in sd):
        return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    return sd


class Model:
    """
    Usage:
        model = Model("trainer-n")            # fresh architecture
        model = Model("runs/exp/weights/best.pt")  # trained checkpoint
        model.train(data="dataset.toml", epochs=100)
        model.val(data="dataset.toml")
        results = model.predict("images/")
        model.export(format="onnx")
    """

    def __init__(self, model: str | Path = "trainer-n", device: str = ""):
        self.ckpt: dict | None = None
        self.checkpoint_path: Path | None = None
        self.overrides: dict = {}
        self.last_run_dir: Path | None = None
        self._fresh_spec: str | None = None
        spec = str(model)
        if spec.endswith(".pt"):
            self.checkpoint_path = Path(spec).resolve()
            self.ckpt = load_checkpoint(spec)
            self.net = model_from_checkpoint(self.ckpt)
            model_sd = self.ckpt["model"]
            if not model_sd:
                # Checkpoint saved with compile=True before the _unwrap fix:
                # ema.ema.state_dict() returned {} for compiled modules.
                # Fall back to the raw (non-EMA) weights which were always unwrapped.
                LOGGER.warning(
                    "checkpoint 'model' state_dict is empty (compiled-EMA bug); "
                    "loading 'raw_model' weights instead"
                )
                model_sd = self.ckpt["raw_model"]
            model_sd = _strip_compiled_prefix(model_sd)
            self.net.load_state_dict(model_sd)
            self.net.names = self.ckpt.get("names", self.net.names)
            self.overrides["model"] = spec
            LOGGER.info(f"loaded checkpoint: {spec} ({self.net.info()})")
        else:
            self.net = DetectionModel(spec)
            self._fresh_spec = spec
            self.overrides["model"] = spec
        if device:
            self.overrides["device"] = device
            self.net = self.net.to(select_device(device))

    @property
    def names(self) -> dict[int, str]:
        return self.net.names

    def _cfg(self, kwargs: dict) -> dict:
        merged = {**self.overrides, **kwargs}
        base_cfg = self.ckpt["cfg"] if self.ckpt else None
        if base_cfg:
            from trainer.detection.config import merge_overrides
            # Start from current defaults so checkpoints created before a new
            # option was introduced still receive that option's safe default.
            cfg = merge_overrides(load_config(), dict(base_cfg))
            return load_config(merge_overrides(cfg, merged))
        return load_config(merged)

    def train(self, data: str, **kwargs) -> dict:
        from trainer.detection.engine.trainer import Trainer
        from trainer.detection.data import parse_dataset_toml

        kwargs["data"] = data
        if kwargs.get("resume"):
            # The resume checkpoint, not this facade's possibly different weights
            # or current defaults, defines the model and training recipe.
            overrides = {k: v for k, v in self.overrides.items() if k != "model"}
            trainer = Trainer(overrides={**overrides, **kwargs})
            result = trainer.train()
            self._load_training_result(trainer)
            return result
        cfg = load_config({**self.overrides, **kwargs})
        parsed = parse_dataset_toml(data)
        if self._fresh_spec is not None:
            # The facade is often created before train(seed=...) is known.
            # Rebuild after seeding so model initialization belongs to the
            # reproducible training state rather than ambient process RNG.
            from trainer.detection.utils.checks import init_seeds

            init_seeds(int(cfg["seed"]), bool(cfg["deterministic"]))
            self.net = DetectionModel(self._fresh_spec)
            self._fresh_spec = None
        target_reg_max = self.net.head.reg_max if self.ckpt is not None else int(cfg["reg_max"])
        target_reg_channels = (self.net.head.reg_channels_override if self.ckpt is not None else int(cfg["reg_channels"]))
        if (self.net.nc != parsed["nc"] or self.net.head.reg_max != target_reg_max or
                self.net.head.reg_channels_override != target_reg_channels):
            old_state = self.net.state_dict()
            self.net = DetectionModel(
                self.net.architecture_spec(), nc=parsed["nc"], ch=self.net.ch, reg_max=target_reg_max,
                reg_channels=target_reg_channels,
            )
            self.net.load_weights(old_state)
        self.net.names = parsed["names"]
        trainer = Trainer(cfg=cfg, model=self.net)
        result = trainer.train()
        self._load_training_result(trainer)
        return result

    def _load_training_result(self, trainer) -> None:
        self.last_run_dir = trainer.run_dir
        self._fresh_spec = None
        selected = trainer.selected_checkpoint
        if selected is not None and selected.exists():
            self.checkpoint_path = selected.resolve()
            self.ckpt = load_checkpoint(selected)
            self.net = model_from_checkpoint(self.ckpt)
            self.net.load_state_dict(_strip_compiled_prefix(self.ckpt["model"] or self.ckpt["raw_model"]))
            self.net.names = self.ckpt["names"]

    def finetune(self, data: str, freeze: int = 10, **kwargs) -> dict:
        """Transfer learning: freeze backbone modules, remap head if nc differs."""
        dataset = load_toml(data) if str(data).endswith(".toml") else {}
        new_nc = len(dataset.get("names", {})) or self.net.nc
        if new_nc != self.net.nc:
            LOGGER.info(f"remapping head: {self.net.nc} -> {new_nc} classes")
            old_state = self.net.state_dict()
            self.net = DetectionModel(
                self.net.architecture_spec(), nc=new_nc, ch=self.net.ch, reg_max=self.net.head.reg_max,
            )
            self.net.load_weights(old_state)
        return self.train(data, freeze=freeze, **kwargs)

    @torch.no_grad()
    def val(self, data: str, **kwargs) -> dict:
        from trainer.detection.data import build_dataloader, parse_dataset_toml
        from trainer.detection.engine.val_writer import setup_file_log, teardown_file_log, write_val_outputs
        from trainer.detection.engine.validator import comparison_metrics, per_class_metrics, per_class_table, validate
        from trainer.detection.trackers import build_tracker
        from trainer.detection.utils.logging import increment_path

        cfg = self._cfg({"data": data, **kwargs})
        selection = "EMA"
        if self.ckpt is not None:
            if cfg["eval_weights"] == "raw":
                if not self.ckpt.get("raw_model"):
                    raise ValueError("raw evaluation requires a complete training checkpoint, not selected.pt")
                state, selection = self.ckpt["raw_model"], "raw"
            else:
                state = self.ckpt.get("model")
                if not state:
                    state, selection = self.ckpt["raw_model"], "raw (legacy EMA fallback)"
            self.net.load_state_dict(_strip_compiled_prefix(state))
        elif cfg["eval_weights"] == "raw":
            raise ValueError("raw evaluation requires a training checkpoint")
        device = select_device(str(cfg["device"]))
        self.net = self.net.to(device)
        parsed = parse_dataset_toml(data)
        if self.net.nc != parsed["nc"]:
            raise ValueError(
                f"model has {self.net.nc} classes but dataset defines {parsed['nc']}; "
                "finetune/remap the detection head before validation"
            )
        loader = build_dataloader(parsed, cfg, "val")

        run_dir = increment_path(Path(cfg["project"]) / cfg["name"], exist_ok=bool(cfg["exist_ok"]))
        self.last_run_dir = run_dir
        LOGGER.info(f"val run dir: {run_dir}")

        tracker = build_tracker(cfg, run_dir)
        tracker.log_params(cfg)
        tracker.set_tags({"model": self.net.model_name, "dataset": str(data), "mode": "val"})

        try:
            log_handler = setup_file_log(run_dir / "logs" / "val.log")
            try:
                result = validate(self.net, loader, cfg, parsed["names"], device=device,
                                  compute_confusion=True, compute_per_image=True,
                                  plot_dir=run_dir / "plots", compute_curves=True,
                                  compute_loss=bool(cfg["val_compute_loss"]))
                if self.checkpoint_path is not None:
                    from trainer.detection.utils.provenance import file_sha256
                    result["checkpoint"] = {"path": str(self.checkpoint_path), "sha256": file_sha256(self.checkpoint_path),
                                            "epoch": int(self.ckpt.get("epoch", -1)) + 1, "selection": selection}
            finally:
                teardown_file_log(log_handler)

            LOGGER.info(per_class_table(result, parsed["names"]))
            write_val_outputs(run_dir, result, parsed["names"], cfg, data, self.net.architecture_spec())
            metrics = {
                "val/mAP50": result["mAP50"], "val/mAP50-95": result["mAP50-95"],
                "val/precision": result["precision"], "val/recall": result["recall"],
            }
            if result["val_loss"] is not None:
                metrics["val/loss"] = result["val_loss"]
            if cfg.get("log_per_class"):
                metrics.update(per_class_metrics(result, parsed["names"]))
            metrics.update(comparison_metrics(result))
            tracker.log_metrics(metrics, step=0)
            for sub in ("cfg", "logs", "plots", "reports"):
                if (run_dir / sub).exists():
                    tracker.log_artifact(run_dir / sub, artifact_path=sub)
            return result
        finally:
            tracker.finish()

    def predict(self, source, conf: float | None = None, embed: bool = False, **kwargs) -> list:
        from trainer.detection.engine.predictor import Predictor

        thresholds = kwargs.pop("thresholds", None)
        batch_size = int(kwargs.pop("batch_size", 16))
        cfg = self._cfg(kwargs)
        device = select_device(str(cfg["device"]))
        self.net = self.net.to(device)
        predictor = Predictor(self.net, cfg, names=self.names, device=device)
        if thresholds:
            if isinstance(thresholds, (str, Path)):
                thresholds = load_toml(thresholds)["thresholds"]
            predictor.set_per_class_thresholds(thresholds)
        return predictor.predict(source, conf=conf, embed=embed, batch_size=batch_size)

    def export(self, format: str = "onnx", **kwargs) -> Path:
        from trainer.detection.engine import exporter

        format = str(format).lower()
        if format not in ("onnx", "tensorrt", "engine", "trt"):
            raise ValueError(f"unknown export format '{format}'")
        out_arg = Path(str(kwargs.pop("out", self.net.model_name)))
        calib_images = kwargs.pop("calib_images", None)
        cfg = self._cfg({"format": format, **kwargs})
        imgsz = int(cfg["imgsz"])
        if format == "onnx" and bool(cfg["half"]):
            raise ValueError("half=true applies to TensorRT export; ONNX export is FP32")
        if format == "onnx":
            onnx_out = out_arg if out_arg.suffix == ".onnx" else Path(f"{out_arg}.onnx")
            engine_out = None
        elif out_arg.suffix == ".engine":
            onnx_out, engine_out = out_arg.with_suffix(".onnx"), out_arg
        elif out_arg.suffix == ".onnx":
            onnx_out, engine_out = out_arg, out_arg.with_suffix(".engine")
        else:
            onnx_out, engine_out = Path(f"{out_arg}.onnx"), Path(f"{out_arg}.engine")
        onnx_path = exporter.export_onnx(
            self.net, imgsz=imgsz, opset=int(cfg["opset"]), dynamic=bool(cfg["dynamic"]),
            half=bool(cfg["half"]), out_path=onnx_out,
        )
        try:
            exporter.verify_onnx_parity(
                self.net, onnx_path, imgsz=imgsz,
                batch_sizes=(1, 2, 4) if bool(cfg["dynamic"]) else (1,),
            )
        except ImportError:
            LOGGER.warning("onnxruntime not installed ([deploy] extra); parity check skipped")
        # Exports contain decoded predictions, not NMS. Persist the host-side contract.
        exporter.write_postprocess_config(onnx_path, cfg, self.names)
        if format == "onnx":
            return onnx_path
        if format in ("tensorrt", "engine", "trt"):
            engine_path = exporter.export_tensorrt(
                onnx_path, half=bool(cfg["half"]), int8=bool(cfg["int8"]),
                workspace_gb=int(cfg["workspace"]), imgsz=imgsz,
                calib_images=calib_images, out_path=engine_out,
            )
            exporter.write_postprocess_config(engine_path, cfg, self.names)
            return engine_path

    def tune_thresholds(self, data: str, save_path: str | Path | None = None, **kwargs) -> dict:
        from trainer.detection.data import build_dataloader, parse_dataset_toml
        from trainer.detection.engine.thresh_tuner import collect_stats, save_thresholds, tune_thresholds

        cfg = self._cfg({"data": data, **kwargs})
        device = select_device(str(cfg["device"]))
        self.net = self.net.to(device)
        parsed = parse_dataset_toml(data)
        if self.net.nc != parsed["nc"]:
            raise ValueError(
                f"model has {self.net.nc} classes but dataset defines {parsed['nc']}; "
                "finetune/remap the detection head before threshold tuning"
            )
        loader = build_dataloader(parsed, cfg, "val")
        stats = collect_stats(self.net, loader, cfg, parsed["nc"], device=device)
        result = tune_thresholds(stats, parsed["nc"], parsed["names"], cfg)
        if save_path:
            save_thresholds(result, save_path)
        return result

    def fuse(self) -> "Model":
        self.net.fuse()
        return self
