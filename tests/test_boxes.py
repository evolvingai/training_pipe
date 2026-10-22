import numpy as np
import torch

from trainer.detection.utils.boxes import (
    bbox2dist,
    bbox_iou,
    box_iou,
    dist2bbox,
    make_anchors,
    xywh2xyxy,
    xyxy2xywh,
)


def test_xywh_xyxy_roundtrip():
    boxes = torch.rand(50, 4) * 100
    boxes[:, 2:] += 1  # ensure positive w/h
    assert torch.allclose(xyxy2xywh(xywh2xyxy(boxes)), boxes, atol=1e-5)


def test_roundtrip_numpy():
    boxes = np.random.rand(10, 4).astype(np.float32) * 100
    boxes[:, 2:] += 1
    assert np.allclose(xyxy2xywh(xywh2xyxy(boxes)), boxes, atol=1e-4)


def test_box_iou_golden():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0], [20.0, 20.0, 30.0, 30.0]])
    iou = box_iou(a, b).squeeze(0)
    assert abs(iou[0].item() - 1.0) < 1e-6
    assert abs(iou[1].item() - 25.0 / 175.0) < 1e-6
    assert iou[2].item() == 0.0


def test_bbox_iou_ciou_le_iou():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[2.0, 2.0, 12.0, 12.0]])
    plain = bbox_iou(a, b)
    ciou = bbox_iou(a, b, ciou=True)
    assert ciou.item() <= plain.item() + 1e-6


def test_dist_bbox_roundtrip():
    anchors = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
    boxes = torch.tensor([[8.0, 7.0, 13.0, 14.0], [18.0, 15.0, 25.0, 24.0]])
    dist = bbox2dist(boxes, anchors, reg_max=16)
    rec = dist2bbox(dist, anchors, xywh=False)
    assert torch.allclose(rec, boxes, atol=1e-5)


def test_make_anchors():
    feats = [torch.zeros(1, 8, 4, 4), torch.zeros(1, 8, 2, 2)]
    anchors, strides = make_anchors(feats, [8, 16])
    assert anchors.shape == (20, 2)
    assert strides.shape == (20, 1)
    assert anchors[0].tolist() == [0.5, 0.5]
    assert strides[0].item() == 8.0
    assert strides[-1].item() == 16.0
