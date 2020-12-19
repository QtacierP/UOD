# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import yaml
import json

args = argparse.ArgumentParser(description='The option of UOD')

args.add_argument('--dataset', type=str, default='gtstb', help='only support GTSDB')
args.add_argument('--data_root_dir', type=str, default='../data', help='Data directory')

args = args.parse_args()

args.data_dir = os.path.join(args.data_root_dir, args.dataset)