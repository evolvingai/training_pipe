"""End-to-end CPU smoke test: synthetic data -> train -> val -> predict ->
threshold tuning -> report -> export. Not a unit test; a fast manual gate."""

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
WORK = ROOT / "runs" / "_smoke"


def make_dataset(root: Path, n_train: int = 16, n_val: int = 8, size: int = 160) -> Path:
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("val", n_val)):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.full((size, size, 3), 30, dtype=np.uint8)
            lines = []
            for _ in range(rng.integers(1, 4)):
                cls = int(rng.integers(0, 2))
                w, h = int(rng.integers(30, 60)), int(rng.integers(30, 60))
                x = int(rng.integers(0, size - w))
                y = int(rng.integers(0, size - h))
                color = (0, 0, 220) if cls == 0 else (0, 220, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
                lines.append(f"{cls} {(x + w / 2) / size:.6f} {(y + h / 2) / size:.6f} "
                             f"{w / size:.6f} {h / size:.6f}")
            cv2.imwrite(str(img_dir / f"{i:03d}.jpg"), img)
            (lbl_dir / f"{i:03d}.txt").write_text("\n".join(lines))
    ds_toml = root / "dataset.toml"
    ds_toml.write_text(
        f'path = "{root}"\ntrain = "images/train"\nval = "images/val"\n\n'
        '[names]\n0 = "red-box"\n1 = "green-box"\n'
    )
    return ds_toml


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    ds = make_dataset(WORK / "data")

    from trainer import Model

    model = Model("trainer-n")
    print("== train ==")
    model.train(str(ds), epochs=2, imgsz=160, batch=4, workers=0, device="cpu",
                project=str(WORK / "runs"), name="exp", mosaic=0.5, warmup_epochs=0.5,
                patience=100, val_period=1, amp=False, deterministic=True)

    print("== val ==")
    result = model.val(str(ds), imgsz=160, batch=4, workers=0, device="cpu")
    print(f"mAP50={result['mAP50']:.3f} mAP50-95={result['mAP50-95']:.3f}")

    print("== predict ==")
    results = model.predict(str(WORK / "data" / "images" / "val"), conf=0.01, imgsz=160,
                            workers=0, device="cpu")
    print(f"predicted {len(results)} images, detections: {[len(r) for r in results]}")

    print("== tune-thresh ==")
    thr = model.tune_thresholds(str(ds), save_path=WORK / "thresholds.toml", imgsz=160,
                                batch=4, workers=0, device="cpu")
    print({k: v["threshold"] for k, v in thr["diagnostics"].items()})

    print("== report ==")
    from trainer.detection.config import load_config
    from trainer.detection.data import parse_dataset_toml
    from trainer.detection.reporting import generate_report
    run_dir = WORK / "runs" / "exp"
    parsed = parse_dataset_toml(ds)
    generate_report(run_dir, result, parsed["names"], load_config({"data": str(ds)}), thresholds=thr)

    print("== export (onnx) ==")
    try:
        out = model.export(format="onnx", imgsz=160, out=str(WORK / "model"))
        print(f"onnx: {out}")
    except Exception as e:  # onnx optional in this environment
        print(f"onnx export skipped/failed: {e}")

    print("== resume smoke ==")
    model2 = Model(str(run_dir / "last.pt"))
    model2.train(str(ds), epochs=3, imgsz=160, batch=4, workers=0, device="cpu",
                 project=str(WORK / "runs"), name="exp", resume=str(run_dir / "last.pt"),
                 mosaic=0.5, warmup_epochs=0.5, patience=100, amp=False)
    print("ALL SMOKE STEPS PASSED")


if __name__ == "__main__":
    main()
