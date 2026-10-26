"""Model facade: the single entry point for train / val / predict / export / tune."""

from __future__ import annotations

from pathlib import Path

import torch

from trainer.detection.config import load_config, load_toml
from trainer.detection.nn.model import DetectionModel
from trainer.detection.utils.checks import select_device
from trainer.detection.utils.logging import LOGGER


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
        self.overrides: dict = {}
        spec = str(model)
        if spec.endswith(".pt"):
            self.ckpt = torch.load(spec, map_location="cpu", weights_only=False)
            self.net = DetectionModel(self.ckpt["model_name"], nc=self.ckpt["nc"])
            self.net.load_state_dict(self.ckpt["model"])
            self.net.names = self.ckpt.get("names", self.net.names)
            self.overrides["model"] = spec
            LOGGER.info(f"loaded checkpoint: {spec} ({self.net.info()})")
        else:
            self.net = DetectionModel(spec)
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
            cfg = dict(base_cfg)
            from trainer.detection.config import merge_overrides
            return merge_overrides(cfg, merged, strict=False)
        return load_config(merged)

    def train(self, data: str, **kwargs) -> dict:
        from trainer.detection.engine.trainer import Trainer

        kwargs["data"] = data
        cfg = load_config({**self.overrides, **kwargs})
        pretrained = self.ckpt is not None
        trainer = Trainer(cfg=cfg, model=self.net if pretrained else None)
        result = trainer.train()
        # reload best weights into this facade (rebuild: nc may have changed)
        best = trainer.run_dir / "weights" / "best.pt"
        if best.exists():
            self.ckpt = torch.load(best, map_location="cpu", weights_only=False)
            self.net = DetectionModel(self.ckpt["model_name"], nc=self.ckpt["nc"])
            self.net.load_state_dict(self.ckpt["model"])
            self.net.names = self.ckpt["names"]
        return result

    def finetune(self, data: str, freeze: int = 10, **kwargs) -> dict:
        """Transfer learning: freeze backbone modules, remap head if nc differs."""
        dataset = load_toml(data) if str(data).endswith(".toml") else {}
        new_nc = len(dataset.get("names", {})) or self.net.nc
        if new_nc != self.net.nc:
            LOGGER.info(f"remapping head: {self.net.nc} -> {new_nc} classes")
            old_state = self.net.state_dict()
            self.net = DetectionModel(self.net.model_name.rsplit("-", 1)[0] + "-" + self.net.scale,
                                      nc=new_nc)
            self.net.load_weights(old_state)
        return self.train(data, freeze=freeze, **kwargs)

    @torch.no_grad()
    def val(self, data: str, **kwargs) -> dict:
        from trainer.detection.data import build_dataloader, parse_dataset_toml
        from trainer.detection.engine.val_writer import setup_file_log, teardown_file_log, write_val_outputs
        from trainer.detection.engine.validator import per_class_table, validate
        from trainer.detection.trackers import build_tracker
        from trainer.detection.utils.logging import increment_path

        cfg = self._cfg({"data": data, **kwargs})
        device = select_device(str(cfg["device"]))
        self.net = self.net.to(device)
        parsed = parse_dataset_toml(data)
        loader = build_dataloader(parsed, cfg, "val")

        run_dir = increment_path(Path(cfg["project"]) / cfg["name"], exist_ok=bool(cfg["exist_ok"]))
        LOGGER.info(f"val run dir: {run_dir}")

        tracker = build_tracker(cfg, run_dir)
        tracker.log_params(cfg)
        tracker.set_tags({"model": self.net.model_name, "dataset": str(data), "mode": "val"})

        log_handler = setup_file_log(run_dir / "logs" / "val.log")
        try:
            result = validate(self.net, loader, cfg, parsed["names"], device=device,
                              compute_confusion=True, compute_per_image=True,
                              plot_dir=run_dir / "plots", compute_curves=True)
        finally:
            teardown_file_log(log_handler)

        LOGGER.info(per_class_table(result, parsed["names"]))
        write_val_outputs(run_dir, result, parsed["names"], cfg, data, self.net.model_name)
        for sub in ("cfg", "logs", "plots", "reports"):
            if (run_dir / sub).exists():
                tracker.log_artifact(run_dir / sub, artifact_path=sub)
        tracker.finish()
        return result

    def predict(self, source, conf: float | None = None, embed: bool = False, **kwargs) -> list:
        from trainer.detection.engine.predictor import Predictor

        cfg = self._cfg(kwargs)
        device = select_device(str(cfg["device"]))
        self.net = self.net.to(device)
        predictor = Predictor(self.net, cfg, names=self.names, device=device)
        thresholds = kwargs.get("thresholds")
        if thresholds:
            if isinstance(thresholds, (str, Path)):
                thresholds = load_toml(thresholds)["thresholds"]
            predictor.set_per_class_thresholds(thresholds)
        return predictor.predict(source, conf=conf, embed=embed)

    def export(self, format: str = "onnx", **kwargs) -> Path:
        from trainer.detection.engine import exporter

        out_stem = kwargs.pop("out", f"{self.net.model_name}")
        cfg = self._cfg({"format": format, **kwargs})
        imgsz = int(cfg["imgsz"])
        onnx_path = exporter.export_onnx(
            self.net, imgsz=imgsz, opset=int(cfg["opset"]), dynamic=bool(cfg["dynamic"]),
            half=bool(cfg["half"]), out_path=f"{out_stem}.onnx",
        )
        try:
            exporter.verify_onnx_parity(self.net, onnx_path, imgsz=imgsz)
        except ImportError:
            LOGGER.warning("onnxruntime not installed ([deploy] extra); parity check skipped")
        if format == "onnx":
            return onnx_path
        if format in ("tensorrt", "engine", "trt"):
            return exporter.export_tensorrt(
                onnx_path, half=bool(cfg["half"]), int8=bool(cfg["int8"]),
                workspace_gb=int(cfg["workspace"]), imgsz=imgsz,
                calib_images=kwargs.get("calib_images"),
            )
        raise ValueError(f"unknown export format '{format}'")

    def tune_thresholds(self, data: str, save_path: str | Path | None = None, **kwargs) -> dict:
        from trainer.detection.data import build_dataloader, parse_dataset_toml
        from trainer.detection.engine.thresh_tuner import collect_stats, save_thresholds, tune_thresholds

        cfg = self._cfg({"data": data, **kwargs})
        device = select_device(str(cfg["device"]))
        self.net = self.net.to(device)
        parsed = parse_dataset_toml(data)
        loader = build_dataloader(parsed, cfg, "val")
        stats = collect_stats(self.net, loader, cfg, parsed["nc"], device=device)
        result = tune_thresholds(stats, parsed["nc"], parsed["names"], cfg)
        if save_path:
            save_thresholds(result, save_path)
        return result

    def fuse(self) -> "Model":
        self.net.fuse()
        return self
