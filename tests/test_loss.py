import torch

from trainer.detection.config import load_config
from trainer.detection.nn.loss import DetectionLoss, TaskAlignedAssigner
from trainer.detection.nn.model import DetectionModel


def make_batch(device="cpu"):
    return {
        "batch_idx": torch.tensor([0, 0, 1]),
        "cls": torch.tensor([0.0, 2.0, 1.0]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.3], [0.2, 0.25, 0.15, 0.2],
                                [0.7, 0.6, 0.4, 0.2]]),
    }


def test_loss_forward_backward():
    cfg = load_config()
    model = DetectionModel("trainer-n", nc=3).train()
    loss_fn = DetectionLoss(model, cfg)
    feats = model(torch.rand(2, 3, 96, 96))
    total, components = loss_fn(feats, make_batch())
    assert total.item() > 0
    assert components.shape == (3,)
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_loss_no_targets():
    cfg = load_config()
    model = DetectionModel("trainer-n", nc=3).train()
    loss_fn = DetectionLoss(model, cfg)
    feats = model(torch.rand(1, 3, 96, 96))
    batch = {"batch_idx": torch.zeros(0, dtype=torch.long), "cls": torch.zeros(0),
             "bboxes": torch.zeros(0, 4)}
    total, components = loss_fn(feats, batch)
    assert torch.isfinite(total)
    assert components[0] == 0  # no box loss without targets


def test_focal_option_changes_loss():
    model = DetectionModel("trainer-n", nc=3).train()
    torch.manual_seed(0)
    x = torch.rand(1, 3, 96, 96)
    feats = model(x)
    bce = DetectionLoss(model, load_config())(feats, make_batch())[1][1]
    focal = DetectionLoss(model, load_config({"cls_loss": "focal"}))([f.detach() for f in feats],
                                                                     make_batch())[1][1]
    assert bce.item() != focal.item()


def test_assigner_prefers_matching_anchor():
    nc = 2
    assigner = TaskAlignedAssigner(topk=3, num_classes=nc)
    anchor_points = torch.tensor([[4.0, 4.0], [12.0, 4.0], [4.0, 12.0], [12.0, 12.0]])
    gt_bboxes = torch.tensor([[[0.0, 0.0, 8.0, 8.0]]])  # covers only anchor 0
    gt_labels = torch.tensor([[[0.0]]])
    mask_gt = torch.ones(1, 1, 1)
    pred_bboxes = torch.tensor([[[0.0, 0.0, 8.0, 8.0],
                                 [8.0, 0.0, 16.0, 8.0],
                                 [0.0, 8.0, 8.0, 16.0],
                                 [8.0, 8.0, 16.0, 16.0]]])
    pred_scores = torch.full((1, 4, nc), 0.5)
    labels, boxes, scores, fg = assigner(pred_scores, pred_bboxes, anchor_points,
                                         gt_labels, gt_bboxes, mask_gt)
    assert fg[0, 0].item() is True or fg[0, 0].item() == 1
    assert not fg[0, 1] and not fg[0, 2] and not fg[0, 3]
    assert labels[0, 0].item() == 0
    assert torch.allclose(boxes[0, 0], gt_bboxes[0, 0])


def test_assigner_empty_gt():
    assigner = TaskAlignedAssigner(topk=3, num_classes=2)
    labels, boxes, scores, fg = assigner(
        torch.rand(1, 4, 2), torch.rand(1, 4, 4), torch.rand(4, 2),
        torch.zeros(1, 0, 1), torch.zeros(1, 0, 4), torch.zeros(1, 0, 1),
    )
    assert not fg.any()
    assert (labels == 2).all()
