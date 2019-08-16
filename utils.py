import os
import torch
import numpy as np
import random

feature_dir = './i3d/rbg'
#feature_tmp = np.load('./i3d/rgb/Hei-Chole1-rgb.npz')

def get_test_cases(feature_name_list, feature_type='rgb', length=512):
    test_cases = []
    for name_item in feature_name_list:
        feature_dic = np.load(os.path.join(feature_dir, name_item))
        print(feature_dic['video_name'])
        feature = list(feature_dic['feature'])
        if (len(feature) < length):
            print('video is shorter than ', length)
            exit()
        else:
            flips_num = len(feature) // length
        flips_feature = []
        for flips_id in range(flips_num):
            flips_feature.append(feature[flips_id * length : (flips_id + 1) * length])
        test_cases.append(flips_feature)
    return test_cases

def get_train_cases(feature_name_list, feature_type='rgb', length=512):
    train_cases = []
    for name_item in feature_name_list:
        feature_dic = np.load(os.path.join(feature_dir, name_item))
        print(feature_dic['video_name'])
        feature = list(feature_dic['feature'])
        if (len(feature) < length):
            print('video is shorter than ', length)
            exit()
        else:
            start_frame = random.randint(0, len(feature) - length)
        feature = feature[start_frame : start_frame+ feature]
        train_cases.append(feature)
    return train_cases
    
def get_gt():
    gt_cases = []

    return gt_cases

