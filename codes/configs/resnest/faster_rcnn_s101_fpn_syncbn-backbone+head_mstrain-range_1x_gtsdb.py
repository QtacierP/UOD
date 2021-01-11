_base_ = './faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_gtsdb.py'
model = dict(
    pretrained='open-mmlab://resnest101',
    backbone=dict(stem_channels=128, depth=101))

load_from = 'http://download.openmmlab.com/mmdetection/v2.0/resnest/faster_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/faster_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20201006_021058-421517f1.pth'