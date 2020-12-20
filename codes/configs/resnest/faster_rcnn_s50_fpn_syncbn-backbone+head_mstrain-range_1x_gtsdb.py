_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_2x_gtsdb.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)


model = dict(
    pretrained='open-mmlab://resnest50',
    backbone=dict(
        type='ResNeSt',
        stem_channels=64,
        depth=50,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch'),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            num_classes=43,
            norm_cfg=norm_cfg)),
)

optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[7])

log_config = dict(interval=50,   hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# # use ResNeSt img_norm
img_norm_cfg = dict(
    mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], to_rgb=True)

load_from = 'http://download.openmmlab.com/mmdetection/v2.0/resnest/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20200926_125502-20289c16.pth'
