import numpy as np
import torch

from trainer.detection.utils.metrics import ConfusionMatrix, DetMetrics, compute_ap, match_predictions


def test_compute_ap_perfect():
    recall = np.array([0.5, 1.0])
    precision = np.array([1.0, 1.0])
    assert abs(compute_ap(recall, precision) - 1.0) < 1e-6


def test_compute_ap_zero():
    assert compute_ap(np.array([0.0]), np.array([0.0])) < 0.02


def test_match_predictions_exact():
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    gt_cls = torch.tensor([0.0])
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    pred_cls = torch.tensor([0.0])
    tp = match_predictions(pred, pred_cls, gt, gt_cls)
    assert tp.all()  # perfect match is a TP at every IoU threshold


def test_match_predictions_wrong_class():
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    tp = match_predictions(torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.tensor([1.0]),
                           gt, torch.tensor([0.0]))
    assert not tp.any()


def test_match_predictions_single_gt_single_match():
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    gt_cls = torch.tensor([0.0])
    preds = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])  # both overlap
    tp = match_predictions(preds, torch.tensor([0.0, 0.0]), gt, gt_cls)
    assert tp[:, 0].sum() == 1  # only one TP at IoU=0.5


def test_detmetrics_perfect_detection():
    m = DetMetrics(nc=2)
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]])
    gt_cls = torch.tensor([0.0, 1.0])
    pred = torch.tensor([
        [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
        [20.0, 20.0, 40.0, 40.0, 0.8, 1.0],
    ])
    m.update(pred, gt, gt_cls)
    r = m.compute()
    assert r["mAP50"] > 0.99
    assert r["mAP50-95"] > 0.99


def test_detmetrics_false_positive_lowers_map():
    m = DetMetrics(nc=1)
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    gt_cls = torch.tensor([0.0])
    pred = torch.tensor([
        [0.0, 0.0, 10.0, 10.0, 0.6, 0.0],
        [50.0, 50.0, 60.0, 60.0, 0.9, 0.0],  # confident FP ranked first
    ])
    m.update(pred, gt, gt_cls)
    r = m.compute()
    assert 0.3 < r["mAP50"] < 0.9


def test_detmetrics_empty():
    m = DetMetrics(nc=3)
    r = m.compute()
    assert r["mAP50"] == 0.0 and r["mAP50-95"] == 0.0


def test_confusion_matrix_counts():
    cm = ConfusionMatrix(nc=2, conf=0.25)
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    gt_cls = torch.tensor([0.0])
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                         [50.0, 50.0, 60.0, 60.0, 0.8, 1.0]])
    cm.update(pred, gt, gt_cls)
    assert cm.matrix[0, 0] == 1        # correct class-0 detection
    assert cm.matrix[1, 2] == 1        # class-1 false positive vs background
    cm.update(torch.zeros((0, 6)), gt, gt_cls)
    assert cm.matrix[2, 0] == 1        # missed gt counts as background prediction
