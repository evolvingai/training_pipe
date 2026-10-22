"""Learning-capability gate: overfit 16 synthetic images; expect mAP50 > 0.6."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.smoke_e2e import WORK, make_dataset  # noqa: E402


def main() -> None:
    ds = WORK / "data" / "dataset.toml"
    if not ds.exists():
        ds = make_dataset(WORK / "data")

    from trainer import Model

    model = Model("trainer-n")
    result = model.train(
        str(ds), epochs=120, imgsz=160, batch=8, workers=0, device="cpu",
        project=str(WORK / "runs"), name="overfit", exist_ok=True,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, translate=0.0, scale=0.0, fliplr=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, erasing=0.0,
        warmup_epochs=2.0, patience=200, val_period=10, amp=False,
        lr0=0.002, optimizer="adamw", close_mosaic=0, early_stop_min_epochs=120,
    )
    map50 = result.get("mAP50", 0.0)
    print(f"FINAL mAP50={map50:.3f} mAP50-95={result.get('mAP50-95', 0.0):.3f}")
    assert map50 > 0.6, f"overfit gate FAILED: mAP50={map50:.3f} <= 0.6"
    print("OVERFIT GATE PASSED")


if __name__ == "__main__":
    main()
