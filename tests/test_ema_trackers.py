import csv

import torch
import torch.nn as nn

from trainer.detection.trackers.base import MultiTracker, Tracker
from trainer.detection.trackers.csv_tracker import CSVTracker
from trainer.detection.utils.ema import ModelEMA


def test_ema_moves_toward_model():
    model = nn.Linear(4, 4)
    ema = ModelEMA(model, decay_max=0.9, warmup_steps=10)
    with torch.no_grad():
        model.weight.add_(1.0)
    before = (ema.ema.weight - model.weight).abs().mean().item()
    for _ in range(20):
        ema.update(model)
    after = (ema.ema.weight - model.weight).abs().mean().item()
    assert after < before


def test_ema_decay_ramp():
    ema = ModelEMA(nn.Linear(2, 2), decay_max=0.9999, warmup_steps=100)
    assert ema._decay() < 0.01  # tracks the model tightly at the start
    ema.step_count = 100_000
    assert abs(ema._decay() - 0.9999) < 1e-6


def test_ema_tracks_model_early():
    """After few steps, EMA must be close to the live model (warmup direction)."""
    model = nn.Linear(4, 4)
    ema = ModelEMA(model, decay_max=0.9999, warmup_steps=2000)
    with torch.no_grad():
        model.weight.fill_(5.0)
    for _ in range(20):
        ema.update(model)
    assert (ema.ema.weight - 5.0).abs().max() < 0.5


def test_ema_state_roundtrip():
    model = nn.Linear(3, 3)
    ema = ModelEMA(model, decay_max=0.99, warmup_steps=5)
    for _ in range(3):
        ema.update(model)
    state = ema.state_dict()
    ema2 = ModelEMA(nn.Linear(3, 3), decay_max=0.5, warmup_steps=1)
    ema2.load_state_dict(state)
    assert ema2.step_count == 3
    assert ema2.decay_max == 0.99
    assert torch.equal(ema2.ema.weight, ema.ema.weight)


def test_csv_tracker_appends_and_dedups(tmp_path):
    path = tmp_path / "results.csv"
    t = CSVTracker(path)
    t.log_metrics({"loss": 1.0}, step=0)
    t.log_metrics({"loss": 0.5, "mAP50": 0.3}, step=1)
    # simulate resume: same steps skipped, new step written
    t2 = CSVTracker(path)
    t2.log_metrics({"loss": 9.9}, step=1)  # duplicate -> ignored
    t2.log_metrics({"loss": 0.4, "mAP50": 0.4}, step=2)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[1]["loss"] == "0.5"


def test_csv_tracker_merges_new_metrics_for_same_step(tmp_path):
    path = tmp_path / "results.csv"
    tracker = CSVTracker(path)
    tracker.log_metrics({"val/AP50_red": 0.2}, step=3)
    tracker.log_metrics({"val/mAP50": 0.4}, step=3)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["val/AP50_red"] == "0.2"
    assert rows[0]["val/mAP50"] == "0.4"


def test_multitracker_swallows_errors():
    class Exploding(Tracker):
        calls = 0

        def log_metrics(self, metrics, step):
            self.calls += 1
            raise RuntimeError("boom")

    calls = []

    class Recording(Tracker):
        def log_metrics(self, metrics, step):
            calls.append(step)

    exploding = Exploding()
    mt = MultiTracker([exploding, Recording()])
    mt.log_metrics({"x": 1.0}, step=5)  # must not raise
    assert calls == [5]
    assert exploding.calls == 1
