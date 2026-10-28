"""Evaluation report in Markdown format."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from trainer.detection.utils.logging import LOGGER
from trainer.detection.utils.plotting import plot_curve


def _md_table(headers: list[str], rows: list[list]) -> str:
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    header = "| " + " | ".join(str(h) for h in headers) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return "\n".join([header, sep, body])


def generate_report(run_dir: str | Path, result: dict, names: dict[int, str], cfg: dict,
                    thresholds: dict | None = None, out_name: str = "report.md",
                    plots_dir: str | Path | None = None) -> Path:
    run_dir = Path(run_dir)
    plots_dir = Path(plots_dir) if plots_dir else run_dir / "plots"
    out_path = run_dir / out_name
    plots_rel = Path(os.path.relpath(plots_dir, out_path.parent))

    lines: list[str] = []

    lines += [
        "# Evaluation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Model:** {cfg.get('model', '?')}  ",
        f"**Dataset:** {cfg.get('data', '?')}  ",
        f"**Image size:** {cfg.get('imgsz', '?')}  ",
        "",
        "## Summary",
        "",
        _md_table(
            ["Metric", "Value"],
            [
                ["mAP50", f"{result['mAP50']:.4f}"],
                ["mAP50-95", f"{result['mAP50-95']:.4f}"],
                ["Precision", f"{result['precision']:.4f}"],
                ["Recall", f"{result['recall']:.4f}"],
            ],
        ),
        "",
    ]
    if result.get("checkpoint"):
        checkpoint = result["checkpoint"]
        lines += ["", f"Checkpoint: {checkpoint['path']} (epoch {checkpoint['epoch']}, "
                  f"{checkpoint['selection']}); SHA256: {checkpoint['sha256']}", ""]
    if result.get("evaluation"):
        meta = result["evaluation"]
        lines += ["", f"Evaluator: {meta['version']}; primary: Trainer; max_det={meta['max_det']}.",
                  f"P/R operating point: {meta['precision_recall']}.",
                  f"Custom crowd policy: {meta['crowd_policy']}.",
                  "Validation loss: " + (str(result["val_loss"]) if result.get("val_loss") is not None
                                         else "not measured"), ""]
    if "pycoco" in result:
        official = result["pycoco"]
        delta = result["pycoco_delta"]
        lines += [
            "## pycocotools Comparison",
            "",
            "Trainer metrics remain the primary validation result. The official evaluator used the same "
            "post-NMS predictions and image IDs.",
            "Official maxDets=[1,10,100]. Custom caps and crowd/ignore handling may differ; "
            "the delta is not solely numerical error.",
            "",
            _md_table(
                ["Metric", "Trainer", "pycocotools", "Delta (Trainer - official)"],
                [
                    ["AP50-95", f"{result['mAP50-95']:.4f}", f"{official['AP']:.4f}",
                     f"{delta['AP50-95']:+.4f}"],
                    ["AP50", f"{result['mAP50']:.4f}", f"{official['AP50']:.4f}",
                     f"{delta['AP50']:+.4f}"],
                ],
            ),
            "",
        ]

    # per-class metrics
    lines += ["## Per-class Metrics", ""]
    rows = []
    low_ap = []
    for c, name in names.items():
        if not result["classes_seen"][c]:
            continue
        p = result["precision_per_class"][c]
        r = result["recall_per_class"][c]
        ap50 = result["ap50_per_class"][c]
        ap = result["ap_per_class"][c]
        rows.append([name, f"{p:.3f}", f"{r:.3f}", f"{ap50:.3f}", f"{ap:.3f}"])
        if ap50 < 0.5:
            low_ap.append(name)
    rows.append(["**all**", f"{result['precision']:.3f}", f"{result['recall']:.3f}",
                 f"{result['mAP50']:.3f}", f"{result['mAP50-95']:.3f}"])
    lines.append(_md_table(["class", "P", "R", "AP50", "AP50-95"], rows))
    if low_ap:
        lines += ["", f"> AP50 < 0.5: {', '.join(low_ap)} — candidates for more data or threshold review."]
    lines.append("")

    # thresholds
    if thresholds:
        lines += ["## Per-class Confidence Thresholds", ""]
        t_rows = []
        diagnostics = thresholds.get("diagnostics", {})
        if diagnostics:
            for name, diag in diagnostics.items():
                t_rows.append([name, diag.get("threshold", "-"), diag.get("best_fitness", "-"),
                               diag.get("precision", "-"), diag.get("recall", "-"),
                               diag.get("status", "-")])
        else:
            values = thresholds.get("thresholds", thresholds)
            t_rows.extend([name, value, "-", "-", "-", "-"] for name, value in values.items())
        lines.append(_md_table(["class", "threshold", "fitness", "P", "R", "status"], t_rows))
        lines.append("")

    # training curves from results.csv
    csv_path = run_dir / "logs" / "results.csv"
    if not csv_path.exists():
        csv_path = run_dir / "results.csv"  # fallback
    if csv_path.exists():
        with csv_path.open() as f:
            rows_csv = list(csv.DictReader(f))
        if rows_csv:
            steps = np.array([float(r["step"]) for r in rows_csv])

            def series(key: str) -> np.ndarray:
                return np.array([float(r[key]) if r.get(key) else np.nan for r in rows_csv])

            keys = rows_csv[0].keys()
            curves = {}
            if "train/box" in keys:
                curves["losses"] = {k: series(k) for k in ("train/box", "train/cls", "train/dfl") if k in keys}
            map_keys = [k for k in ("val/mAP50", "val/mAP50-95") if k in keys]
            if map_keys:
                curves["mAP"] = {k: series(k) for k in map_keys}
            curve_axes = {"losses": ("Epoch", "Loss"), "mAP": ("Epoch", "mAP")}
            if curves:
                lines += ["## Training Curves", ""]
                for title, ys in curves.items():
                    xl, yl = curve_axes.get(title, ("Epoch", ""))
                    png = plot_curve(steps, ys, plots_dir / f"curve_{title}.png",
                                     title=title, xlabel=xl, ylabel=yl)
                    lines.append(f"![{title}]({plots_rel / png.name})")
                lines.append("")

    # plots
    plot_files = [
        ("PR_curve.png", "Precision-Recall Curve"),
        ("F1_curve.png", "F1-Confidence Curve"),
        ("confusion_matrix.png", "Confusion Matrix"),
        ("val_batch0_labels.jpg", "Batch 0 — Labels"),
        ("val_batch0_pred.jpg", "Batch 0 — Predictions"),
    ]
    existing = [(f, t) for f, t in plot_files if (plots_dir / f).exists()]
    if existing:
        lines += ["## Plots", ""]
        for fname, title in existing:
            lines.append(f"### {title}")
            lines.append(f"![{title}]({plots_rel / fname})")
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info(f"report written: {out_path}")
    return out_path
