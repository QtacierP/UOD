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
                10: 'no overtaking (trucks) (prohibitory)',
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
                41: 'restriction ends (overtaking) (other)',
                42: 'restriction ends (overtaking (trucks)) (other)'}

dataset_type = 'CoCoDataset'
data_root = 'data/GTSDB/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
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
        img_scale=(2048, 1024),
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
for label in class_map.keys():
    classes.append(class_map[label])

classes = tuple(classes)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='../data/GTSDB/annotations/train/train.json',
        pipeline=train_pipeline),

    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='../data/GTSDB/annotations/test/test.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='../data/GTSDB/annotations/test/test.json',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
