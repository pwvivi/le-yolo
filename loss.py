# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.atss import ATSSAssigner, generate_anchors
from .metrics import bbox_iou, probiou, bbox_mpdiou, bbox_inner_iou, bbox_focaler_iou, bbox_inner_mpdiou, bbox_focaler_mpdiou, wasserstein_loss, WiseIouLoss
from .tal import bbox2dist

import math

class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.nwd_loss = False
        self.iou_ratio = 0.5
        
        self.use_wiseiou = False
        
        if self.use_wiseiou:
            self.wiou_loss = WiseIouLoss(ltype='WIoU', monotonous=False, inner_iou=False, focaler_iou=False)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, mpdiou_hw=None):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        if self.use_wiseiou:
            wiou = self.wiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], ret_iou=False, ratio=0.7, d=0.0, u=0.95).unsqueeze(-1)
            # wiou = self.wiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], ret_iou=False, ratio=0.7, d=0.0, u=0.95, **{'scale':0.0}).unsqueeze(-1) # Wise-ShapeIoU,Wise-Inner-ShapeIoU,Wise-Focaler-ShapeIoU
            # wiou = self.wiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], ret_iou=False, ratio=0.7, d=0.0, u=0.95, **{'mpdiou_hw':mpdiou_hw[fg_mask]}).unsqueeze(-1) # Wise-MPDIoU,Wise-Inner-MPDIoU,Wise-Focaler-MPDIoU
            loss_iou = (wiou * weight).sum() / target_scores_sum
        else:
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            # iou = bbox_inner_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True, ratio=0.7)
            # iou = bbox_mpdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, mpdiou_hw=mpdiou_hw[fg_mask])
            # iou = bbox_inner_mpdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, mpdiou_hw=mpdiou_hw[fg_mask], ratio=0.7)
            # iou = bbox_focaler_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True, d=0.0, u=0.95)
            # iou = bbox_focaler_mpdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, mpdiou_hw=mpdiou_hw[fg_mask], d=0.0, u=0.95)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
                
        if self.nwd_loss:
            nwd = wasserstein_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            nwd_loss = ((1.0 - nwd) * weight).sum() / target_scores_sum
            loss_iou = self.iou_ratio * loss_iou +  (1 - self.iou_ratio) * nwd_loss

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)

