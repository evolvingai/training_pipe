import torch

from trainer.detection.utils.nms import nms, non_max_suppression


def test_nms_suppresses_overlaps():
    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],   # heavy overlap with box 0
        [50.0, 50.0, 60.0, 60.0],
    ])
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_thres=0.5)
    assert keep.tolist() == [0, 2]


def test_nms_keeps_all_disjoint():
    boxes = torch.tensor([[0.0, 0, 5, 5], [10.0, 10, 15, 15], [20.0, 20, 25, 25]])
    scores = torch.tensor([0.5, 0.9, 0.7])
    keep = nms(boxes, scores, iou_thres=0.5)
    assert sorted(keep.tolist()) == [0, 1, 2]
    assert keep.tolist()[0] == 1  # score-descending


def _make_prediction(nc=3, entries=()):
    """entries: (cx, cy, w, h, cls, conf)."""
    pred = torch.zeros(1, 4 + nc, max(len(entries), 1))
    for i, (cx, cy, w, h, cls, conf) in enumerate(entries):
        pred[0, :4, i] = torch.tensor([cx, cy, w, h])
        pred[0, 4 + cls, i] = conf
    return pred


def test_batched_nms_confidence_filter():
    pred = _make_prediction(entries=[(50, 50, 20, 20, 0, 0.9), (100, 100, 10, 10, 1, 0.1)])
    out = non_max_suppression(pred, conf_thres=0.25)[0]
    assert out.shape == (1, 6)
    assert int(out[0, 5]) == 0


def test_batched_nms_class_aware():
    # same location, different classes: both survive class-aware NMS
    pred = _make_prediction(entries=[(50, 50, 20, 20, 0, 0.9), (50, 50, 20, 20, 1, 0.8)])
    out = non_max_suppression(pred, conf_thres=0.25)[0]
    assert out.shape[0] == 2
    agn = non_max_suppression(pred, conf_thres=0.25, agnostic=True)[0]
    assert agn.shape[0] == 1


def test_per_class_thresholds():
    pred = _make_prediction(entries=[(50, 50, 20, 20, 0, 0.30), (100, 100, 10, 10, 1, 0.30)])
    per_class = torch.tensor([0.5, 0.2, 0.9])
    out = non_max_suppression(pred, conf_thres=0.25, per_class_conf=per_class)[0]
    assert out.shape[0] == 1
    assert int(out[0, 5]) == 1  # only class 1 passes its 0.2 threshold


def test_max_det_cap():
    entries = [(10.0 * i, 10.0 * i, 5, 5, 0, 0.9) for i in range(20)]
    pred = _make_prediction(entries=entries)
    out = non_max_suppression(pred, conf_thres=0.25, max_det=5)[0]
    assert out.shape[0] == 5
