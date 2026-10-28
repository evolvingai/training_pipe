"""Detection loss: task-aligned assignment (TAL), BCE/focal cls, CIoU + DFL box."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.detection.utils.boxes import bbox2dist, bbox_iou, dist2bbox, make_anchors, xywh2xyxy


class TaskAlignedAssigner(nn.Module):
    """Assign predictions to ground truths by alignment = score^alpha * IoU^beta.

    For each gt: candidates are anchors whose center lies inside the gt box;
    the top-k by alignment become positives. Anchors claimed by multiple gts go
    to the gt with the highest IoU.
    """

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 0.5, beta: float = 6.0,
                 eps: float = 1e-9, metric: str = "iou", diagnostics: bool = False):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        if metric not in ("iou", "ciou"):
            raise ValueError("assignment metric must be iou or ciou")
        self.metric, self.diagnostics = metric, diagnostics
        self.last_stats = {}

    @torch.no_grad()
    def forward(self, pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        """All shapes batch-first:
        pred_scores (B, A, nc) sigmoid scores; pred_bboxes (B, A, 4) xyxy (grid units);
        anchor_points (A, 2); gt_labels (B, N, 1); gt_bboxes (B, N, 4) xyxy (grid units);
        mask_gt (B, N, 1) valid-gt mask.
        Returns target_labels (B, A), target_bboxes (B, A, 4), target_scores (B, A, nc),
        fg_mask (B, A) bool.
        """
        bs, n_max, _ = gt_bboxes.shape
        na = pred_bboxes.shape[1]
        device = pred_bboxes.device
        if self.diagnostics:
            self.last_stats = {"gt_count": mask_gt.sum().detach(), "gts_with_positives": torch.zeros((), device=device)}
        if n_max == 0:
            return (
                torch.full((bs, na), self.num_classes, device=device, dtype=torch.long),
                torch.zeros((bs, na, 4), device=device),
                torch.zeros((bs, na, self.num_classes), device=device),
                torch.zeros((bs, na), dtype=torch.bool, device=device),
            )

        # anchors inside gt boxes: (B, N, A)
        lt = anchor_points[None, None] - gt_bboxes[:, :, None, :2]
        rb = gt_bboxes[:, :, None, 2:] - anchor_points[None, None]
        in_gt = torch.cat([lt, rb], -1).amin(-1) > self.eps

        # alignment metric (B, N, A)
        ious = (self._pairwise_iou(gt_bboxes, pred_bboxes) if self.metric == "iou" else
                bbox_iou(gt_bboxes.unsqueeze(2), pred_bboxes.unsqueeze(1), ciou=True)).clamp_(0)
        idx = gt_labels.long().squeeze(-1).clamp_(0, self.num_classes - 1)  # (B, N)
        scores = pred_scores.permute(0, 2, 1)  # (B, nc, A)
        cls_score = scores.gather(1, idx.unsqueeze(-1).expand(-1, -1, na))  # (B, N, A)
        align = cls_score.pow(self.alpha) * ious.pow(self.beta)
        align = align * in_gt * mask_gt

        # top-k per gt
        topk = min(self.topk, na)
        topk_vals, topk_idx = align.topk(topk, dim=-1)
        topk_mask = topk_vals > self.eps
        mask_pos = torch.zeros_like(align, dtype=torch.bool)
        mask_pos.scatter_(-1, topk_idx, topk_mask)
        mask_pos = mask_pos & in_gt & mask_gt.bool()

        # resolve multi-gt anchors by IoU
        fg_counts = mask_pos.sum(1)  # (B, A)
        if (fg_counts > 1).any():
            multi = (fg_counts.unsqueeze(1) > 1).expand(-1, n_max, -1)
            best_gt = ious.masked_fill(~mask_pos, -1).argmax(1)  # eligible candidates only
            best_mask = torch.zeros_like(mask_pos)
            best_mask.scatter_(1, best_gt.unsqueeze(1), True)
            mask_pos = torch.where(multi, best_mask & mask_pos, mask_pos)
        fg_mask = mask_pos.any(1)  # (B, A)
        if self.diagnostics:
            self.last_stats["gts_with_positives"] = (mask_pos.any(-1) & mask_gt.squeeze(-1).bool()).sum().detach()
        assigned_gt = mask_pos.float().argmax(1)  # (B, A)

        batch_idx = torch.arange(bs, device=device).unsqueeze(-1)
        target_labels = idx[batch_idx, assigned_gt]  # (B, A)
        target_bboxes = gt_bboxes[batch_idx, assigned_gt]  # (B, A, 4)
        target_labels = torch.where(fg_mask, target_labels, torch.full_like(target_labels, self.num_classes))

        # normalized target scores: alignment scaled so max per gt equals max IoU
        align_pos = align * mask_pos
        pos_align_max = align_pos.amax(-1, keepdim=True)
        pos_iou_max = (ious * mask_pos).amax(-1, keepdim=True)
        norm_align = (align_pos * pos_iou_max / (pos_align_max + self.eps)).amax(1)  # (B, A)
        target_scores = torch.zeros((bs, na, self.num_classes), device=device)
        valid = fg_mask
        target_scores[batch_idx.expand(-1, na)[valid], torch.arange(na, device=device).expand(bs, -1)[valid],
                      target_labels[valid]] = norm_align[valid]
        return target_labels, target_bboxes, target_scores, fg_mask

    @staticmethod
    def _pairwise_iou(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """(B, N, 4) x (B, A, 4) -> (B, N, A) IoU."""
        gt_ = gt.unsqueeze(2)  # (B, N, 1, 4)
        pr = pred.unsqueeze(1)  # (B, 1, A, 4)
        inter = (torch.min(gt_[..., 2:], pr[..., 2:]) - torch.max(gt_[..., :2], pr[..., :2])).clamp_(0).prod(-1)
        a_gt = (gt_[..., 2:] - gt_[..., :2]).prod(-1)
        a_pr = (pr[..., 2:] - pr[..., :2]).prod(-1)
        return inter / (a_gt + a_pr - inter + 1e-9)


class DetectionLoss:
    def __init__(self, model, cfg: dict):
        head = model.head
        self.nc = head.nc
        self.reg_max = head.reg_max
        self.no = head.no
        self.stride = head.stride
        self.device = next(model.parameters()).device
        self.diagnostics = bool(cfg.get("loss_diagnostics"))
        self.last_diagnostics = {}
        self.assigner = TaskAlignedAssigner(topk=int(cfg["tal_topk"]), num_classes=self.nc,
                                            metric=cfg.get("assignment_metric", "iou"), diagnostics=self.diagnostics)
        self.gain_box = float(cfg["box"])
        self.gain_cls = float(cfg["cls"])
        self.gain_dfl = float(cfg["dfl"])
        self.cls_loss = str(cfg.get("cls_loss", "bce"))
        self.focal_gamma = float(cfg.get("focal_gamma", 1.5))
        self.label_smoothing = float(cfg.get("label_smoothing", 0.0))
        if self.label_smoothing:
            raise ValueError("label_smoothing must be 0: smoothing quality targets is not supported")
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)
        self._anchor_cache: dict = {}

    def _cls_term(self, pred_logits: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred_logits, target_scores, reduction="none")
        if self.cls_loss == "focal":
            p = pred_logits.sigmoid()
            pt = target_scores * p + (1 - target_scores) * (1 - p)
            bce = bce * (1 - pt).pow(self.focal_gamma)
        return bce.sum()

    def _dfl_term(self, pred_dist: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
        """Distribution focal loss: CE against the two nearest integer bins."""
        tl = target_dist.long()
        tr = tl + 1
        wl = tr.float() - target_dist
        wr = 1 - wl
        loss_l = F.cross_entropy(pred_dist, tl.clamp_(0, self.reg_max - 1), reduction="none") * wl
        loss_r = F.cross_entropy(pred_dist, tr.clamp_(0, self.reg_max - 1), reduction="none") * wr
        return (loss_l + loss_r).mean(-1, keepdim=True)

    def __call__(self, feats: list[torch.Tensor], batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (total loss * batch size, detached components [box, cls, dfl])."""
        device = feats[0].device
        if self.proj.device != device or self.proj.dtype != feats[0].dtype:
            self.proj = self.proj.to(device=device, dtype=feats[0].dtype)
        bs = feats[0].shape[0]
        cat = torch.cat([f.view(bs, self.no, -1) for f in feats], 2)
        pred_dist, pred_logits = cat.split((self.reg_max * 4, self.nc), 1)
        pred_dist = pred_dist.permute(0, 2, 1).contiguous()  # (B, A, 4*reg_max)
        pred_logits = pred_logits.permute(0, 2, 1).contiguous()  # (B, A, nc)

        cache_key = (tuple(f.shape[-2:] for f in feats), feats[0].dtype)
        if cache_key not in self._anchor_cache:
            a, s = make_anchors(feats, self.stride, 0.5)
            img_h = feats[0].shape[2] * int(self.stride[0])
            img_w = feats[0].shape[3] * int(self.stride[0])
            sv = torch.tensor([img_w, img_h, img_w, img_h], device=device)
            self._anchor_cache[cache_key] = (a, s, sv)
        anchors, strides, scale_vec = self._anchor_cache[cache_key]
        na = anchors.shape[0]

        # decode boxes in grid units
        dist = pred_dist.view(bs, na, 4, self.reg_max).softmax(3) @ self.proj  # (B, A, 4)
        pred_bboxes = dist2bbox(dist, anchors.unsqueeze(0), xywh=False)  # grid units

        # build padded gt tensors (grid units of the concatenated scales)
        batch_idx = batch["batch_idx"]
        counts = torch.bincount(batch_idx.to(device), minlength=bs)
        n_max = int(counts.max().item()) if counts.numel() else 0
        gt_labels = torch.zeros((bs, n_max, 1), device=device)
        gt_bboxes = torch.zeros((bs, n_max, 4), device=device)
        mask_gt = torch.zeros((bs, n_max, 1), device=device)
        if n_max:
            b_idx = batch_idx.to(device)
            offset = torch.cat([torch.zeros(1, device=device, dtype=torch.long),
                                counts.cumsum(0)[:-1]])
            pos = torch.arange(b_idx.shape[0], device=device) - offset[b_idx]
            gt_labels[b_idx, pos, 0] = batch["cls"].to(device)
            gt_bboxes[b_idx, pos] = xywh2xyxy(batch["bboxes"].to(device) * scale_vec)
            mask_gt[b_idx, pos, 0] = 1.0
        pred_bboxes_px = pred_bboxes * strides  # (B, A, 4) pixels

        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_logits.detach().sigmoid(), pred_bboxes_px.detach(), anchors * strides,
            gt_labels, gt_bboxes, mask_gt,
        )
        target_scores_sum = target_scores.sum().clamp_(min=1)
        if self.diagnostics:
            self.last_diagnostics = {**self.assigner.last_stats, "positive_anchors": fg_mask.sum().detach(),
                                     "clipped_dfl_sides": torch.zeros((), device=device),
                                     "total_dfl_sides": fg_mask.sum().detach() * 4}

        loss_cls = self._cls_term(pred_logits, target_scores) / target_scores_sum

        loss_box = torch.zeros(1, device=device)
        loss_dfl = torch.zeros(1, device=device)
        if fg_mask.any():
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            iou = bbox_iou(pred_bboxes_px[fg_mask], target_bboxes[fg_mask], ciou=True)
            loss_box = ((1.0 - iou).unsqueeze(-1) * weight).sum() / target_scores_sum
            # DFL target distances in grid units of each anchor's stride
            strides_b = strides.unsqueeze(0).expand(bs, -1, -1)[fg_mask]  # (n, 1)
            anchors_b = anchors.unsqueeze(0).expand(bs, -1, -1)[fg_mask]  # (n, 2)
            target_dist = bbox2dist(target_bboxes[fg_mask] / strides_b, anchors_b, self.reg_max)
            if self.diagnostics:
                boxes_grid = target_bboxes[fg_mask] / strides_b
                raw_dist = torch.cat((anchors_b - boxes_grid[:, :2], boxes_grid[:, 2:] - anchors_b), -1)
                positive_boxes = target_bboxes[fg_mask]
                area = (positive_boxes[:, 2:] - positive_boxes[:, :2]).prod(-1)
                self.last_diagnostics.update({
                    "clipped_dfl_sides": (raw_dist > self.reg_max - 1).sum().detach(),
                    "positive_small": (area < 32**2).sum().detach(),
                    "positive_medium": ((area >= 32**2) & (area < 96**2)).sum().detach(),
                    "positive_large": (area >= 96**2).sum().detach(),
                    "assignment_overlap_sum": bbox_iou(pred_bboxes_px[fg_mask], target_bboxes[fg_mask],
                                                        ciou=self.assigner.metric == "ciou").clamp_min(0).sum().detach(),
                })
                overlap = bbox_iou(pred_bboxes_px[fg_mask], positive_boxes, ciou=self.assigner.metric == "ciou").clamp_min(0)
                for size, selected in (("small", area < 32**2), ("medium", (area >= 32**2) & (area < 96**2)),
                                       ("large", area >= 96**2)):
                    self.last_diagnostics[f"overlap_{size}_sum"] = overlap[selected].sum().detach()
            pd = pred_dist[fg_mask].view(-1, 4, self.reg_max).permute(0, 2, 1)  # (n, reg_max, 4)
            loss_dfl = (self._dfl_term(pd, target_dist) * weight).sum() / target_scores_sum

        total = (self.gain_box * loss_box + self.gain_cls * loss_cls + self.gain_dfl * loss_dfl)
        components = torch.stack([
            (self.gain_box * loss_box).detach().squeeze(),
            (self.gain_cls * loss_cls).detach().squeeze(),
            (self.gain_dfl * loss_dfl).detach().squeeze(),
        ])
        return total.squeeze() * bs, components
