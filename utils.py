import os
import torch
import numpy as np
import random
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
import subprocess
from config import *
import pandas as pd

def get_test_cases(feature_name_list, feature_type, length):
    feature_dir = '../i3d'
    feature_dir = os.path.join(feature_dir, feature_type)
    test_cases = []
    for name_item in feature_name_list:
        feature_npz = np.load(os.path.join(feature_dir, name_item))
        print(feature_dir + '/' + name_item, ' for test')
        feature = feature_npz['feature'].tolist()
        feature = feature[0]
        if (len(feature) < length):
            print('video length is ', len(feature))
            print('video is shorter than ', length)
            exit()
        else:
            flips_num = len(feature) // length
        flips_feature = []
        for flips_id in range(flips_num):
            flips_feature.append(feature[flips_id * length : (flips_id + 1) * length])
        test_cases.append(flips_feature)
    return test_cases

def get_train_cases(feature_name_list, feature_type, length):
    feature_dir = '../i3d'
    feature_dir = os.path.join(feature_dir, feature_type)
    train_cases = []
    for name_item in feature_name_list:
        feature_npz = np.load(os.path.join(feature_dir, name_item))
        print(feature_dir + '/' + name_item, ' for train')
        feature = list(feature_npz['feature'])
        feature = feature[0]
        if (len(feature) < length):
            print('video length is ', len(feature))
            print('video is shorter than ', length)
            exit()
        else:
            start_frame = random.randint(0, len(feature) - length)
        feature = feature[start_frame : (start_frame + length)]
        train_cases.append(feature)
    return train_cases

def get_gt(feature_name):
    gt_cases = []
    tmp = feature_name.split('-')
    name = '-'.join([tmp[0], tmp[1]]) + '_'
    gt_dir = '../../Annotations/'
    gt_paths = [os.path.join(gt_dir, i) for i in os.listdir(gt_dir) if (i.endswith('.csv') and i.startswith(name))]
    for gt_path in gt_paths:
        print(gt_path)
        gt_data = pd.read_csv(gt_path)
        print(gt_data.shape)
        #print(type(gt_data))
        gt_cases.append(gt_data)
    return gt_cases


def get_phase_error(pred_phase, gt_phase):
    # pred_phase: numpy, frames x 4
    criterion = nn.CrossEntropyLoss()
    gt_phase = torch.from_numpy(gt_phase)
    loss = criterion(pred_phase, gt_phase)
    return loss


def get_instrument_error(pred_instrument, gt_instrument):
    criterion = nn.BCEWithLogitsLoss()
    gt_instrument = torch.from_numpy(gt_instrument)
    loss = criterion(pred_instrument, gt_instrument)
    return loss


def get_action_error(pred_action, gt_action):
    criterion = nn.BCEWithLogitsLoss()
    gt_action = torch.from_numpy(gt_action)
    loss = criterion(pred_action, gt_action)
    return loss


def num2name(num_list, feature_type):
    name_list = []
    for item in num_list:
        s = 'Hei-Chole' + item
        s = s + '-' + feature_type + '.npz'
        name_list.append(s)
    return name_list
