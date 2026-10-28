"""Mutation-based hyperparameter evolution with explicit stopping criteria."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

from trainer.detection.config import dump as toml_dump
from trainer.detection.config import load_config
from trainer.detection.utils.logging import LOGGER, increment_path

# name: (mutation sigma fraction, min, max)
SEARCH_SPACE: dict[str, tuple[float, float, float]] = {
    "lr0": (0.3, 1e-5, 0.1),
    "lrf": (0.3, 0.001, 0.5),
    "momentum": (0.1, 0.7, 0.98),
    "weight_decay": (0.3, 0.0, 0.001),
    "warmup_epochs": (0.3, 0.0, 5.0),
    "box": (0.3, 1.0, 20.0),
    "cls": (0.3, 0.2, 4.0),
    "dfl": (0.3, 0.4, 6.0),
    "hsv_h": (0.3, 0.0, 0.1),
    "hsv_s": (0.3, 0.0, 0.9),
    "hsv_v": (0.3, 0.0, 0.9),
    "translate": (0.3, 0.0, 0.5),
    "scale": (0.3, 0.0, 0.9),
    "mosaic": (0.3, 0.0, 1.0),
    "mixup": (0.3, 0.0, 0.5),
    "copy_paste": (0.3, 0.0, 0.5),
    "fliplr": (0.3, 0.0, 1.0),
}


def mutate(hyp: dict, rng: random.Random, mutation_p: float = 0.8) -> dict:
    """Gaussian mutation of ~80% of keys, clamped to bounds."""
    out = dict(hyp)
    while True:
        changed = False
        for key, (sigma, lo, hi) in SEARCH_SPACE.items():
            if rng.random() > mutation_p:
                continue
            base = float(out[key])
            span = hi - lo
            value = base + rng.gauss(0, sigma) * (span if base == 0 else abs(base) + 0.05 * span)
            out[key] = round(min(max(value, lo), hi), 6)
            if out[key] != hyp[key]:
                changed = True
        if changed:
            return out


TUNE_METRICS = ("mAP50-95", "mAP50", "precision", "recall", "f1", "val_loss")


def _trial_fitness(result: dict, metric: str) -> float:
    """Extract a scalar fitness (higher = better) from a validation result dict."""
    if metric == "val_loss":
        return -float(result["val_loss"])
    if metric == "f1":
        p = float(result.get("precision", 0.0))
        r = float(result.get("recall", 0.0))
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    if metric not in result:
        raise ValueError(
            f"tune_metric '{metric}' not found in validation results. "
            f"Available: {TUNE_METRICS}"
        )
    return float(result[metric])


class Tuner:
    """Evolution loop: mutate best-known hyps, run short trainings, keep the best.

    Stops on: max iterations, no-improvement patience, max trials, or time budget.
    """

    def __init__(self, overrides: dict | None = None):
        self.cfg = load_config(overrides)
        metric = str(self.cfg.get("tune_metric", "mAP50-95"))
        if metric not in TUNE_METRICS:
            raise ValueError(f"unknown tune_metric '{metric}'. Choose from: {TUNE_METRICS}")
        self.metric = metric
        self.rng = random.Random(int(self.cfg["seed"]))
        self.dir = increment_path(Path(self.cfg["project"]) / f"{self.cfg['name']}_tune")
        self.history: list[tuple[float, dict]] = []

    def run(self, time_budget_s: float | None = None) -> dict:
        from trainer.detection.engine.trainer import Trainer

        cfg = self.cfg
        iterations = int(cfg["tune_iterations"])
        patience = int(cfg["tune_patience"])
        min_delta = float(cfg["tune_min_delta"])
        min_trials = int(cfg["tune_min_trials"])
        max_trials = int(cfg["tune_max_trials"])

        LOGGER.info(f"tuning metric: {self.metric}")
        base_hyp = {k: cfg[k] for k in SEARCH_SPACE}
        best_fitness = -math.inf
        best_hyp = dict(base_hyp)
        no_improve = 0
        t0 = time.time()

        for trial in range(min(iterations, max_trials)):
            hyp = mutate(best_hyp, self.rng) if self.history else dict(base_hyp)
            trial_cfg = dict(cfg)
            trial_cfg.update(hyp)
            trial_cfg.update({
                "epochs": int(cfg["tune_epochs"]),
                "project": str(self.dir),
                "name": f"trial{trial}",
                "exist_ok": True,
                "patience": 10**9,  # no early stop inside short trials
                "mlflow": False,
                "resume": False,
            })
            try:
                result = Trainer(cfg=trial_cfg).train()
                fitness = _trial_fitness(result, self.metric)
            except Exception as e:  # noqa: BLE001 - a bad trial must not kill the tuner
                LOGGER.warning(f"trial {trial} failed: {e}")
                fitness = -math.inf
            self.history.append((fitness, hyp))
            improved = fitness > best_fitness + min_delta
            if improved:
                best_fitness = fitness
                best_hyp = hyp
                no_improve = 0
            else:
                no_improve += 1
            fitness_text = f"{fitness:.4f}" if math.isfinite(fitness) else "failed"
            best_text = f"{best_fitness:.4f}" if math.isfinite(best_fitness) else "none"
            LOGGER.info(f"tune trial {trial + 1} [{self.metric}]: fitness={fitness_text} "
                        f"best={best_text} (no improvement x{no_improve})")
            if math.isfinite(best_fitness):
                self._save(best_hyp, best_fitness)
            if trial + 1 >= min_trials and no_improve >= patience:
                LOGGER.info(f"tuner stopped: {patience} trials without improvement")
                break
            if time_budget_s and time.time() - t0 > time_budget_s:
                LOGGER.info("tuner stopped: time budget exhausted")
                break
        if not math.isfinite(best_fitness):
            raise RuntimeError("all tuning trials failed; inspect the trial warnings above")
        return best_hyp

    def _save(self, best_hyp: dict, best_fitness: float) -> None:
        toml_dump({"fitness": best_fitness, "tune_metric": self.metric, **best_hyp},
                  self.dir / "best_hyp.toml")
