"""Write structured evaluation outputs into a val run directory.

Directory layout produced:
    run_dir/
        cfg/
            args.toml       resolved val config
            model.toml      model architecture config
            dataset.toml    dataset config copy
        logs/
            val_results.json  global + per-class metrics
            val_results.csv   tabular, one row per class + global
            val.log           full INFO+ log captured during validate()
        plots/
            confusion_matrix.png
            PR_curve.png
            F1_curve.png
            val_batch{n}_labels.jpg   (written by validator.py)
            val_batch{n}_pred.jpg     (written by validator.py)
        reports/
            report.md
"""

from __future__ import annotations

import csv
import json
import logging
import tomllib
from pathlib import Path

import cv2
import numpy as np

from trainer.detection.config import dumps as toml_dumps
from trainer.detection.utils.image import imwrite_rgb
from trainer.detection.utils.logging import LOGGER
from trainer.detection.utils.plotting import plot_curve


# ---------------------------------------------------------------------------
# File log management
# ---------------------------------------------------------------------------

def setup_file_log(log_path: Path) -> logging.FileHandler:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    LOGGER.addHandler(handler)
    return handler


def teardown_file_log(handler: logging.FileHandler) -> None:
    handler.flush()
    handler.close()
    LOGGER.removeHandler(handler)


# ---------------------------------------------------------------------------
# cfg/
# ---------------------------------------------------------------------------

def _base_model_name(model_name: str) -> str:
    """'trainer-n' -> 'trainer', 'trainer-csp-m' -> 'trainer-csp'."""
    parts = model_name.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) == 1 and parts[1].isalpha():
        return parts[0]
    return model_name


def write_cfg(run_dir: Path, cfg: dict, data_path: str, model_name: str | dict) -> None:
    cfg_dir = run_dir / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Preserve lists/tables; unsupported types fail instead of silently disappearing.
    args = {k: v for k, v in cfg.items() if v is not None}
    (cfg_dir / "args.toml").write_text(toml_dumps(args), encoding="utf-8")

    # model.toml — architecture config
    from trainer.detection.config import find_model_cfg
    model_cfg_written = False
    if isinstance(model_name, dict):
        (cfg_dir / "model.toml").write_text(toml_dumps(model_name), encoding="utf-8")
        model_cfg_written = True
    else:
        for name_try in [model_name, _base_model_name(model_name)]:
            try:
                model_cfg_path = find_model_cfg(name_try)
                with model_cfg_path.open("rb") as f:
                    model_cfg = tomllib.load(f)
                (cfg_dir / "model.toml").write_text(toml_dumps(model_cfg), encoding="utf-8")
                model_cfg_written = True
                break
            except FileNotFoundError:
                continue
    if not model_cfg_written:
        (cfg_dir / "model.toml").write_text(toml_dumps({"model": model_name}), encoding="utf-8")

    # dataset.toml — verbatim copy of data file
    dp = Path(data_path)
    if dp.is_file():
        (cfg_dir / "dataset.toml").write_bytes(dp.read_bytes())
    else:
        (cfg_dir / "dataset.toml").write_text(toml_dumps({"data": data_path}), encoding="utf-8")


# ---------------------------------------------------------------------------
# logs/
# ---------------------------------------------------------------------------

def write_logs(run_dir: Path, result: dict, names: dict[int, str]) -> None:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    per_class: dict[str, dict] = {}
    for c, name in names.items():
        if result["classes_seen"][c]:
            per_class[name] = {
                "precision": round(float(result["precision_per_class"][c]), 6),
                "recall": round(float(result["recall_per_class"][c]), 6),
                "AP50": round(float(result["ap50_per_class"][c]), 6),
                "AP50-95": round(float(result["ap_per_class"][c]), 6),
            }

    json_data = {
        "global": {
            "mAP50": round(result["mAP50"], 6),
            "mAP50-95": round(result["mAP50-95"], 6),
            "precision": round(result["precision"], 6),
            "recall": round(result["recall"], 6),
            "val_loss": (round(float(result["val_loss"]), 6) if result.get("val_loss") is not None else None),
        },
        "per_class": per_class,
        "evaluation": result.get("evaluation", {}),
        "checkpoint": result.get("checkpoint"),
    }
    if "pycoco" in result:
        json_data["pycoco"] = {
            key: round(float(value), 6) for key, value in result["pycoco"].items()
        }
        json_data["pycoco_delta"] = {
            key: round(float(value), 6) for key, value in result["pycoco_delta"].items()
        }
    (logs_dir / "val_results.json").write_text(
        json.dumps(json_data, indent=2), encoding="utf-8"
    )

    fieldnames = ["class", "precision", "recall", "AP50", "AP50-95"]
    rows: list[dict] = []
    for c, name in names.items():
        if result["classes_seen"][c]:
            rows.append({
                "class": name,
                "precision": f"{result['precision_per_class'][c]:.4f}",
                "recall": f"{result['recall_per_class'][c]:.4f}",
                "AP50": f"{result['ap50_per_class'][c]:.4f}",
                "AP50-95": f"{result['ap_per_class'][c]:.4f}",
            })
    rows.append({
        "class": "all",
        "precision": f"{result['precision']:.4f}",
        "recall": f"{result['recall']:.4f}",
        "AP50": f"{result['mAP50']:.4f}",
        "AP50-95": f"{result['mAP50-95']:.4f}",
    })
    with (logs_dir / "val_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if "per_image_stats" in result:
        stats = result["per_image_stats"]
        per_img_rows = stats.per_image_class_rows()
        with (logs_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "class", "TP", "FP", "FN", "TN", "FP confused with"])
            writer.writerows(per_img_rows)
        with (logs_dir / "fp_details.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "predicted class", "matched ground-truth class"])
            writer.writerows(stats.fp_detail_rows())


# ---------------------------------------------------------------------------
# plots/
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(matrix: np.ndarray, names: dict[int, str], out_path: Path) -> None:
    nc = len(names)
    n = nc + 1  # include background
    labels = [names[c] for c in sorted(names)] + ["bg"]

    cell = max(40, min(80, 1600 // n))
    canvas_h = cell * n + cell  # extra row for column headers
    canvas_w = cell * n + cell  # extra col for row headers
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    max_val = max(int(matrix.max()), 1)
    font, fscale, thick = cv2.FONT_HERSHEY_SIMPLEX, max(0.3, cell / 120), 1

    for r in range(n):
        for ci in range(n):
            val = int(matrix[r, ci])
            ratio = val / max_val
            # diagonal = blue tint (TP), off-diagonal = red tint (errors)
            if r == ci:
                color = (int(255 * (1 - ratio * 0.8)), int(255 * (1 - ratio * 0.8)), 255)
            else:
                color = (255, int(255 * (1 - ratio * 0.8)), int(255 * (1 - ratio * 0.8)))
            x0 = cell + ci * cell
            y0 = cell + r * cell
            cv2.rectangle(canvas, (x0, y0), (x0 + cell - 1, y0 + cell - 1), color, -1)
            if val > 0:
                txt = str(val)
                (tw, th), _ = cv2.getTextSize(txt, font, fscale, thick)
                cv2.putText(canvas, txt,
                            (x0 + (cell - tw) // 2, y0 + (cell + th) // 2),
                            font, fscale, (0, 0, 0), thick, cv2.LINE_AA)

    # column headers (predicted class) and row headers (actual class)
    for i, lbl in enumerate(labels):
        short = lbl[:6]
        (tw, th), _ = cv2.getTextSize(short, font, fscale, thick)
        # top header
        cv2.putText(canvas, short,
                    (cell + i * cell + (cell - tw) // 2, th + 4),
                    font, fscale, (40, 40, 40), thick, cv2.LINE_AA)
        # left header
        cv2.putText(canvas, short,
                    (2, cell + i * cell + (cell + th) // 2),
                    font, fscale, (40, 40, 40), thick, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite_rgb(out_path, canvas)


def write_plots(run_dir: Path, result: dict, names: dict[int, str]) -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if "confusion" in result:
        _plot_confusion_matrix(result["confusion"], names, plots_dir / "confusion_matrix.png")

    if "pr_recall" in result:
        plot_curve(result["pr_recall"], {"mean": result["pr_precision"]},
                   plots_dir / "PR_curve.png", title="Precision-Recall",
                   xlabel="Recall", ylabel="Precision")

    if "f1_conf" in result:
        plot_curve(result["f1_conf"], {"mean": result["f1_mean"]},
                   plots_dir / "F1_curve.png", title="F1-Confidence",
                   xlabel="Confidence", ylabel="F1")


# ---------------------------------------------------------------------------
# reports/
# ---------------------------------------------------------------------------

def write_report(run_dir: Path, result: dict, names: dict[int, str], cfg: dict) -> None:
    from trainer.detection.reporting.report import generate_report
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    generate_report(run_dir, result, names, cfg,
                    plots_dir=run_dir / "plots",
                    out_name="reports/report.md")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def write_val_outputs(run_dir: Path, result: dict, names: dict[int, str],
                      cfg: dict, data_path: str, model_name: str | dict) -> None:
    write_cfg(run_dir, cfg, data_path, model_name)
    write_logs(run_dir, result, names)
    write_plots(run_dir, result, names)
    write_report(run_dir, result, names, cfg)
    LOGGER.info(f"evaluation outputs saved → {run_dir}")
