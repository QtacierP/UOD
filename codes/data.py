import mmcv
from gen_coco_label import class_map

def make_data_loader(args):
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
            ann_file='{}/train/train.json'.format(args.data_dir)),
        val=dict(
            type=dataset_type,
            classes=classes,
            ann_file='{}/val/val.json'.format(args.data_dir)),
        test=dict(
            type=dataset_type,
            classes=classes,
            ann_file='{}/test/test.json'.format(args.data_dir)))
    return data

def make_process(args):
    pass