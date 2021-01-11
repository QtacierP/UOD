_base_ = './faster_rcnn_r50_fpn_2x_gtsdb.py'
model = dict(
    pretrained='open-mmlab://resnext101_64x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        style='pytorch'))

load_from = 'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_2x_coco/faster_rcnn_x101_64x4d_fpn_2x_coco_20200512_161033-5961fa95.pth'
