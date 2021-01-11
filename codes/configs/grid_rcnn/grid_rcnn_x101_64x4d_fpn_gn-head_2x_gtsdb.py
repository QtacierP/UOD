_base_ = './grid_rcnn_r50_fpn_gn-head_2x_gtsdb.py'

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
        style='pytorch'))


# optimizer
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=3665,
    warmup_ratio=1.0 / 80,
    step=[17, 23])
total_epochs = 150

class_map = {   0 : 'speed limit 20 (prohibitory)',
                1: 'speed limit 30 (prohibitory)',
                2: 'speed limit 50 (prohibitory)',
                3: 'speed limit 60 (prohibitory)',
                4: 'speed limit 70 (prohibitory)',
                5: 'speed limit 80 (prohibitory)',
                6: 'restriction ends 80 (other)',
                7: 'speed limit 100 (prohibitory)',
                8: 'speed limit 120 (prohibitory)',
                9: 'no overtaking (prohibitory)',
                10: 'no overtaking trucks (prohibitory)',
                11: 'priority at next intersection (danger)',
                12: 'priority road (other)',
                13: 'give way (other)',
                14: 'stop (other)',
                15: 'no traffic both ways (prohibitory)',
                16: 'no trucks (prohibitory)',
                17: 'no entry (other)',
                18: 'danger (danger)',
                19: 'bend left (danger)',
                20: 'bend right (danger)',
                21: 'bend (danger)',
                22: 'uneven road (danger)',
                23: 'slippery road (danger)',
                24: 'road narrows (danger)',
                25: 'construction (danger)',
                26: 'traffic signal (danger)',
                27: 'pedestrian crossing (danger)',
                28: 'school crossing (danger)',
                29: 'cycles crossing (danger)',
                30: 'snow (danger)',
                31: 'animals (danger)',
                32: 'restriction ends (other)',
                33: 'go right (mandatory)',
                34: 'go left (mandatory)',
                35: 'go straight (mandatory)',
                36: 'go right or straight (mandatory)',
                37: 'go left or straight (mandatory)',
                38: 'keep right (mandatory)',
                39: 'keep left (mandatory)',
                40: 'roundabout (mandatory)',
                41: 'restriction ends overtaking (other)',
                42: 'restriction ends overtaking-trucks (other)'}

dataset_type = 'CoCoDataset'
data_root = 'data/GTSDB/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', img_scale=[(1360, 800), (680, 400)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1360, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
dataset_type = 'CocoDataset'
classes = []
for index in class_map.keys():
    label = class_map[index]
    label, superlabel = label.split('(')
    print(label)
    label = label[:-1]
    superlabel = superlabel[:-1]
    classes.append(label)

classes = tuple(classes)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='../data/GTSDB/annotations/train/train.json',
        pipeline=train_pipeline),

    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='../data/GTSDB/annotations/val/val.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='../data/GTSDB/annotations/test/test.json',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

load_from = 'http://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco_20200204-ec76a754.pth'