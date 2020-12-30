import torch.nn as nn
import torch
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES, NECKS, HEADS, LOSSES
from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared2FCBBoxHead, ConvFCBBoxHead
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmcv.runner import auto_fp16, force_fp32
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.models.losses import accuracy
from mmdet.models.losses.utils import weight_reduce_loss


@LOSSES.register_module()
class UncertaintyLoss(nn.Module):
    def __init__(self,  loss_weight=1.0):
        super(UncertaintyLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                uncertainty, weight, avg_factor=None, reduction_override='mean'):
        loss = self.calculate_loss(pred, target, uncertainty)
        return loss

    def calculate_loss(self, pred, target, log_sigma):
        sigma = torch.exp(-log_sigma)
        mse = F.mse_loss(pred, target)
        return sigma * mse + log_sigma


@HEADS.register_module()
class UncertaintyBoxHead(ConvFCBBoxHead):
    def __init__(self, num_uncertainty_fcs=0, num_uncertainty_convs=0, 
    with_uncertainty=True, loss_bbox=dict(
                     type='UncertaintyLoss', loss_weight=1.0),  *args,**kwargs):
        kwargs.setdefault('with_avg_pool', True)
        super(UncertaintyBoxHead, self).__init__(loss_bbox=loss_bbox, **kwargs)
        self.with_uncertainty = with_uncertainty
        self.num_uncertainty_fcs = num_uncertainty_fcs
        self.num_uncertainty_convs = num_uncertainty_convs
        self.uncertainty_convs, self.uncertainty_fcs, self.uncertainty_last_dim \
            = self._add_conv_fc_branch(self.num_uncertainty_convs, self.num_uncertainty_fcs, self.shared_out_channels)
        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_uncertainty_fcs == 0:
                self.uncertainty_last_dim *= self.roi_feat_area

        if self.with_uncertainty:
            out_dim_uncertainty = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_uncertainty = nn.Linear(self.uncertainty_last_dim, out_dim_uncertainty)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches

        x_cls = x
        x_reg = x
        x_uncertainty = x

        # Classification 
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        # Regression 
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        
        # Uncertainty
        for conv in self.uncertainty_convs:
            x_uncertainty = conv(x_uncertainty)
        if x_uncertainty.dim() > 2:
            if self.with_avg_pool:
                x_uncertainty = self.avg_pool(x_uncertainty)
            x_uncertainty = x_uncertainty.flatten(1)
        for fc in self.uncertainty_fcs:
            x_uncertainty = self.relu(fc(x_uncertainty))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        log_sigma = self.fc_uncertainty(x_uncertainty) if self.with_uncertainty else None

        return cls_score, bbox_pred, log_sigma

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             log_sigma,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    log_sigma,
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses


@HEADS.register_module()
class UncertaintyRoIHead(StandardRoIHead):
    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, log_sigma = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, log_sigma=log_sigma)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], bbox_results['log_sigma'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

@HEADS.register_module()
class UncertaintyShared2FCBBoxHead(UncertaintyBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(UncertaintyShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
