import json
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import glob
from tqdm import tqdm


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


original_data_dir = '../data/original_GTSDB/FullIJCNN2013'
original_gt_dir = '../data/original_GTSDB/FullIJCNN2013/gt.txt'

def process_gt(str_gt):
    gt = {}
    for one_gt in str_gt:
        img_name, bbox_1, bbox_2, bbox_3, bbox_4, label = one_gt.split(';')
        gt[img_name] = {}
        gt[img_name]['bbox'] = [int(bbox_1), int(bbox_2), int(bbox_3), int(bbox_4)]
        gt[img_name]['label'] = int(label)
    return gt

def covert_to_coco(gt, split_list, output_image_dir, output_annotation_dir, base_annotation_id=0):
    annotations = {}
    annotations['images'] = []
    annotations['annotations'] = []
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    annotation_dir, _ = os.path.split(output_annotation_dir)
    if not os.path.exists(annotation_dir):
        os.makedirs(annotation_dir)
    for annotation_id, one_gt in tqdm(enumerate(gt)):
        img_name, bbox_1, bbox_2, bbox_3, bbox_4, label = one_gt.split(';')

        id = img_name.split('.')[0]
        bbox_1, bbox_2, bbox_3, bbox_4 = int(bbox_1), int(bbox_2), int(bbox_3), int(bbox_4)

        img_path = os.path.join(original_data_dir, img_name)

        if img_path not in split_list:
            continue
        img = plt.imread(img_path)
        h, w, c = img.shape
        one_annotation = {}
        one_annotation['image'] = {}
        one_annotation['image']['file_name'] = img_path
        one_annotation['image']['height'] = h
        one_annotation['image']['width'] = w
        one_annotation['image']['id'] = id
        one_annotation['annotations'] = {}
        x, y = bbox_1, bbox_2
        bbox_h = bbox_4 - y
        bbox_w = bbox_3 - x
        one_annotation['annotations']['bbox'] = [x, y, bbox_w, bbox_h]
        one_annotation['annotations']['image_id'] = id
        one_annotation['annotations']['category_id'] = int(label)
        one_annotation['annotations']['area'] = bbox_h * bbox_w
        one_annotation['annotations']['iscrowd'] = 0
        one_annotation['annotations']['id'] = base_annotation_id + annotation_id
        annotations['images'].append(one_annotation['image'])
        annotations['annotations'].append(one_annotation['annotations'])
    annotations['categories'] = []
    for i in class_map.keys():
        label = class_map[i]
        info = label.split('(')
        label = info[0][:-1]
        superlabel = info[1][:-1]
        annotations['categories'].append({'id': i, 'name': label, "supercategory": superlabel})
    with open(output_annotation_dir, 'w') as f:
        json.dump(annotations, f)
    return annotation_id + base_annotation_id + 1, annotations




if __name__ == '__main__':
    random.seed(0)
    img_list = glob.glob(os.path.join(original_data_dir, '*.ppm'))

    gt = np.loadtxt(original_gt_dir, dtype=str)
    train_list, test_list = train_test_split(img_list, test_size=0.2)
    train_list, val_list = train_test_split(train_list, test_size=0.2)
    id, _ = covert_to_coco(gt, train_list, '../data/GTSDB/images/train', '../data/GTSDB/annotations/train/train.json')
    id, _ = covert_to_coco(gt, val_list, '../data/GTSDB/images/val', '../data/GTSDB/annotations/val/val.json', id)
    id, _ = covert_to_coco(gt, test_list, '../data/GTSDB/images/test', '../data/GTSDB/annotations/test/test.json', id)


