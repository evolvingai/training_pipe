"""CLI: trainer <command> [key=value ...]

Commands: train, val, predict, finetune, export, tune, tune-thresh, report,
viz-augment, bench-data, bench-model, version.
"""

from __future__ import annotations

import sys
from pathlib import Path

from trainer.detection.config import load_config, load_toml
from trainer.detection.utils.logging import LOGGER

COMMANDS = ("train", "val", "predict", "finetune", "export", "tune", "tune-thresh",
            "report", "viz-augment", "bench-data", "bench-model", "version", "help")

USAGE = """trainer - in-house real-time detection training framework

usage: trainer <command> [key=value ...]

commands:
  train        train a model              trainer train data=ds.toml model=trainer-n epochs=100
  val          evaluate a checkpoint      trainer val model=runs/exp/weights/best.pt data=ds.toml
  predict      run inference              trainer predict model=best.pt source=images/ conf=0.4
  finetune     transfer-learn             trainer finetune model=best.pt data=new.toml freeze=10
  export       export onnx / tensorrt     trainer export model=best.pt format=onnx
  tune         evolve hyperparameters     trainer tune data=ds.toml tune_iterations=30
  tune-thresh  per-class conf thresholds  trainer tune-thresh model=best.pt data=ds.toml
  report       markdown evaluation report trainer report model=best.pt data=ds.toml
  viz-augment  visualize augmentations    trainer viz-augment data=ds.toml n=16
  bench-data   dataloader throughput      trainer bench-data data=ds.toml
  bench-model  profile model sizes        trainer bench-model model=trainer scales=n,s,m,l,x
  version      print version
"""


def parse_kv(args: list[str]) -> dict:
    out: dict = {}
    for arg in args:
        if "=" not in arg:
            raise SystemExit(f"expected key=value, got '{arg}'")
        key, value = arg.split("=", 1)
        low = value.strip().lower()
        if low in ("true", "false"):
            out[key] = low == "true"
        else:
            try:
                out[key] = int(value) if "." not in value and "e" not in low else float(value)
            except ValueError:
                out[key] = value
    return out


def _model_from(overrides: dict):
    from trainer.detection.api import Model

    spec = str(overrides.pop("model", "trainer-n"))
    device = str(overrides.pop("device", ""))
    return Model(spec, device=device)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] in ("help", "-h", "--help"):
        print(USAGE)
        return 0
    command, *rest = argv
    if command not in COMMANDS:
        print(f"unknown command '{command}'\n\n{USAGE}")
        return 2
    overrides = parse_kv(rest)

    if command == "version":
        from trainer.detection import __version__
        print(__version__)
        return 0

    if command == "train":
        data = overrides.pop("data", None)
        if not data:
            raise SystemExit("train requires data=<dataset.toml>")
        model = _model_from(overrides)
        model.train(data, **overrides)
        return 0

    if command == "finetune":
        data = overrides.pop("data", None)
        if not data:
            raise SystemExit("finetune requires data=<dataset.toml>")
        freeze = int(overrides.pop("freeze", 10))
        model = _model_from(overrides)
        model.finetune(data, freeze=freeze, **overrides)
        return 0

    if command == "val":
        data = overrides.pop("data", None)
        if not data:
            raise SystemExit("val requires data=<dataset.toml>")
        model = _model_from(overrides)
        model.val(data, **overrides)
        return 0

    if command == "predict":
        source = overrides.pop("source", None)
        if not source:
            raise SystemExit("predict requires source=<images>")
        save_dir = Path(str(overrides.pop("save_dir", "runs/predict")))
        model = _model_from(overrides)
        results = model.predict(source, **overrides)
        save_dir.mkdir(parents=True, exist_ok=True)
        from trainer.detection.utils.image import imwrite_rgb
        for r in results:
            out = save_dir / Path(r.path).name
            imwrite_rgb(out, r.plot())
            LOGGER.info(f"{r.path}: {len(r)} detections -> {out}")
        return 0

    if command == "export":
        model = _model_from(overrides)
        fmt = str(overrides.pop("format", "onnx"))
        model.export(format=fmt, **overrides)
        return 0

    if command == "tune":
        from trainer.detection.engine.tuner import Tuner
        if "data" not in overrides:
            raise SystemExit("tune requires data=<dataset.toml>")
        best = Tuner(overrides).run()
        LOGGER.info(f"best hyperparameters: {best}")
        return 0

    if command == "tune-thresh":
        data = overrides.pop("data", None)
        if not data:
            raise SystemExit("tune-thresh requires data=<dataset.toml>")
        save = overrides.pop("save", "thresholds.toml")
        model = _model_from(overrides)
        result = model.tune_thresholds(data, save_path=save, **overrides)
        for name, diag in result["diagnostics"].items():
            LOGGER.info(f"{name:<20s} thr={diag['threshold']:<8} fitness={diag['best_fitness']:<8} "
                        f"[{diag['status']}]")
        return 0

    if command == "report":
        data = overrides.pop("data", None)
        if not data:
            raise SystemExit("report requires data=<dataset.toml>")
        run_dir = overrides.pop("run_dir", None)
        thresholds_path = overrides.pop("thresholds", None)
        model = _model_from(overrides)
        out_dir = Path(run_dir) if run_dir else Path("runs/report")
        result = model.val(
            data,
            project=str(out_dir.parent),
            name=out_dir.name,
            exist_ok=True,
            **overrides,
        )
        from trainer.detection.data import parse_dataset_toml
        from trainer.detection.reporting import generate_report
        if thresholds_path:
            parsed = parse_dataset_toml(data)
            thresholds = load_toml(thresholds_path)
            actual_dir = model.last_run_dir or out_dir
            cfg = model._cfg({"data": data, **overrides})
            generate_report(
                actual_dir,
                result,
                parsed["names"],
                cfg,
                thresholds=thresholds,
                plots_dir=actual_dir / "plots",
                out_name="reports/report.md",
            )
        return 0

    if command == "viz-augment":
        return _viz_augment(overrides)

    if command == "bench-data":
        return _bench_data(overrides)

    if command == "bench-model":
        return _bench_model(overrides)

    return 2


def _viz_augment(overrides: dict) -> int:
    import cv2
    import numpy as np
    from trainer.augmentation import build_pipeline
    from trainer.detection.data import parse_dataset_toml
    from trainer.detection.data.build import _resolve_split
    from trainer.detection.data.dataset import DetectionDataset
    from trainer.detection.utils.image import imwrite_rgb
    from trainer.detection.utils.plotting import draw_boxes, make_grid

    data = overrides.pop("data", None)
    if not data:
        raise SystemExit("viz-augment requires data=<dataset.toml>")
    n = int(overrides.pop("n", 16))
    out = Path(str(overrides.pop("out", "augment_preview.png")))
    cfg = load_config(overrides, )
    parsed = parse_dataset_toml(data)
    paths, _ = _resolve_split(parsed["train"], parsed["root"])
    pipeline = build_pipeline(cfg, train=True, seed=int(cfg["seed"]))
    ds = DetectionDataset(paths, imgsz=int(cfg["imgsz"]), transforms=pipeline)
    tiles = []
    for i in range(n):
        sample = ds.load_sample(i % len(ds))
        sample = pipeline(sample)
        img = draw_boxes(sample["image"], sample["boxes"], sample["cls"], parsed["names"])
        tiles.append(cv2.resize(img, (320, 320)))
    imwrite_rgb(out, make_grid(tiles, cols=int(np.ceil(np.sqrt(n)))))
    LOGGER.info(f"augmentation preview: {out}")
    return 0


def _bench_data(overrides: dict) -> int:
    import time
    from trainer.detection.data import build_dataloader, parse_dataset_toml

    data = overrides.pop("data", None)
    if not data:
        raise SystemExit("bench-data requires data=<dataset.toml>")
    n_batches = int(overrides.pop("n", 50))
    cfg = load_config({"data": data, **overrides})
    parsed = parse_dataset_toml(data)
    loader = build_dataloader(parsed, cfg, "train")
    t0 = time.time()
    seen = 0
    for i, batch in enumerate(loader):
        seen += batch["img"].shape[0]
        if i + 1 >= n_batches:
            break
    dt = time.time() - t0
    LOGGER.info(f"dataloader: {seen} images in {dt:.1f}s = {seen / dt:.1f} img/s "
                f"(batch={cfg['batch']}, workers={cfg['workers']}, cache={cfg['cache']})")
    return 0


def _bench_model(overrides: dict) -> int:
    from trainer.detection.engine.benchmark import benchmark_models, write_benchmark
    from trainer.detection.utils.checks import select_device

    model_base = str(overrides.pop("model", "trainer"))
    suffix = model_base.rsplit("-", 1)[-1]
    if suffix in {"n", "s", "m", "l", "x", "nano", "small", "medium", "large", "xlarge"}:
        model_base = model_base.rsplit("-", 1)[0]
    scales = [value.strip() for value in str(overrides.pop("scales", "n,s,m,l,x")).split(",") if value.strip()]
    warmup = int(overrides.pop("warmup", 10))
    iterations = int(overrides.pop("iterations", 50))
    out = Path(str(overrides.pop("out", "model_benchmark.json")))
    cfg = load_config(overrides)
    result = benchmark_models(
        model_base, scales, imgsz=int(cfg["imgsz"]), batch=int(cfg["batch"]),
        device=select_device(str(cfg["device"])), warmup=warmup, iterations=iterations,
    )
    write_benchmark(result, out)
    LOGGER.info(f"model benchmark: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
