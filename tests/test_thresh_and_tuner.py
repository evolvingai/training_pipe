import random

import numpy as np
import pytest

from trainer.detection.config import load_config
from trainer.detection.engine.thresh_tuner import _fitness, tune_thresholds
from trainer.detection.engine.tuner import SEARCH_SPACE, Tuner, mutate


def make_stats(n=200, seed=0):
    """Synthetic stats: class 0 well-separated (TPs high-conf), class 1 mixed."""
    rng = np.random.default_rng(seed)
    conf, cls, tp, gt = [], [], [], []
    for _ in range(n):
        # class 0: TPs at conf ~0.9, FPs at ~0.1
        is_tp = rng.random() < 0.5
        conf.append(rng.normal(0.9 if is_tp else 0.1, 0.03))
        cls.append(0)
        tp.append(is_tp)
        if is_tp:
            gt.append(0)
        # class 1: everything at the same confidence -> inseparable
        is_tp1 = rng.random() < 0.3
        conf.append(0.5 + rng.normal(0, 0.01))
        cls.append(1)
        tp.append(is_tp1)
        if is_tp1:
            gt.append(1)
    return [{
        "conf": np.clip(np.array(conf), 0.01, 0.99),
        "cls": np.array(cls),
        "tp": np.array(tp, dtype=bool),
        "gt_cls": np.array(gt),
    }]


def test_thresholds_separate_good_class():
    cfg = load_config()
    stats = make_stats()
    out = tune_thresholds(stats, nc=2, names={0: "good", 1: "bad"}, cfg=cfg)
    good = out["diagnostics"]["good"]
    assert good["status"] == "ok"
    assert 0.1 < good["threshold"] < 0.9
    assert good["best_fitness"] > 0.9


def test_thresholds_flag_missing_class():
    cfg = load_config()
    out = tune_thresholds([], nc=1, names={0: "empty"}, cfg=cfg)
    assert out["diagnostics"]["empty"]["status"] == "no_data"


def test_fitness_variants():
    p = np.array([0.9, 0.5])
    r = np.array([0.4, 0.8])
    f1 = _fitness(p, r, "f1")
    assert f1.shape == (2,)
    floor = _fitness(p, r, "precision>=0.8")
    assert floor[0] == 0.4 and floor[1] == -1.0
    rfloor = _fitness(p, r, "recall>=0.5")
    assert rfloor[0] == -1.0 and rfloor[1] == 0.5


def test_mutate_respects_bounds():
    cfg = load_config()
    base = {k: cfg[k] for k in SEARCH_SPACE}
    rng = random.Random(0)
    for _ in range(50):
        out = mutate(base, rng)
        assert out != base
        for key, (_, lo, hi) in SEARCH_SPACE.items():
            assert lo <= out[key] <= hi


def test_all_failed_tuning_trials_raise(monkeypatch, tmp_path):
    from trainer.detection.engine.trainer import Trainer

    monkeypatch.setattr(Trainer, "train", lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    tuner = Tuner({
        "data": "unused.toml", "project": str(tmp_path), "name": "failed",
        "tune_iterations": 2, "tune_max_trials": 2, "tune_min_trials": 1,
    })
    with pytest.raises(RuntimeError, match="all tuning trials failed"):
        tuner.run()
