import os
import torch
import numpy as np
import random
import torch.nn as nn
from config import *
import ipdb
import matplotlib.pyplot as plt
import torch.nn.functional as F
import inspect, re

sig_f = nn.Sigmoid()

"""
name principle: acc/loss + train/eval/test + phase/action/instrument
"""


def get_test_cases(feature_name_list, feature_type, length):
    """
    video clip features, no need to multiply i3d_time
    :param feature_name_list: 
    :param feature_type: 
    :param length: 
    :return: test_cases: n item, each item is a list, a video's whole feature
        clip_nums: n item, each item is the number of video feature
    """
    fea_dir = os.path.join(feature_dir, feature_type)
    test_cases = []
    clip_nums = []

    for name_item in feature_name_list:
        feature_npz = np.load(os.path.join(fea_dir, name_item))
        # print(fea_dir + '/' + name_item, ' for test')
        feature = feature_npz['feature'].tolist()
        feature = feature[0]
        if len(feature) < length:
            print('video length is ', len(feature))
            print('video is shorter than ', length)
            exit()
        clip_num = len(feature) // length
        clip_nums.append(clip_num)

        clip_feature = []
        for clip_id in range(clip_num):
            clip_feature.append(
                feature[clip_id * length: (clip_id + 1) * length])
        test_cases.append(clip_feature)

    return test_cases, clip_nums


def get_train_case(feature_name_list, feature_type, length):
    name = feature_name_list[random.randint(0, len(feature_name_list) - 1)]
    fea_dir = os.path.join(feature_dir, feature_type)
    feature_npz = np.load(os.path.join(fea_dir, name))
    feature = feature_npz['feature'].tolist()
    feature = feature[0]
    if len(feature) < length:
        print('video length is ', len(feature))
        print('video is shorter than ', length)
        exit()
    start_frame = random.randint(0, len(feature) - length)
    data = feature[start_frame: start_frame + length]
    return data, name, start_frame


# ground truth extraction, need to multiply i3d_time
def get_train_gt(feature_name, start_frame, length, times=sample_step):
    feature_name = feature_name.lower()
    tmp = feature_name.split('-')
    name = '-'.join([tmp[0], tmp[1]])+'_'
    gt_paths = [os.path.join(gt_dir, file) for file in os.listdir(gt_dir) if (file.endswith('.csv') and file.startswith(name))]
    gt_phase = None
    gt_instrument = None
    gt_action = None
    for gt_path in gt_paths:
        tmp1 = np.loadtxt(gt_path, delimiter=",")
        tmp1 = np.array(tmp1)
        gt_data = tmp1[0:, 1:]
        gt_data = gt_data[start_frame: start_frame+(times*length), 0:]

        if gt_path.endswith('phase.csv'):
            gt_phase = gt_data
        elif gt_path.endswith('instrument.csv'):
            gt_instrument = gt_data[0:, 0:7]
        elif gt_path.endswith('instrument_detailed.csv'):
            gt_instrument_detailed = gt_data[0:, 0:21]
        elif gt_path.endswith('action.csv'):
            gt_action = gt_data
        elif gt_path.endswith('action_detailed.csv'):
            gt_action_detailed = gt_data

    if gt_phase is None or gt_instrument is None or gt_action is None:
        print(gt_paths)
        exit(1)
    return gt_phase, gt_instrument, gt_action


# print(get_train_gt('Hei-Chole1-flow.npz', 0, 512))


def get_test_gt(feature_name, clip_num, length):
    """
    :param feature_name: 
    :param clip_num: 
    :param length: 
    :return: a list with clip_num items, each item is a dictionary, 
    each key is a gt, each value is the clip annotation
    """
    gts = []
    # if feature_name == num2name([2], 'flow')[0]:
    #    ipdb.set_trace()
    for i in range(clip_num):
        gt = {}
        gt['gt_phase'], gt['gt_instrument'], gt['gt_action'] = \
            get_train_gt(feature_name, i * length * sample_step, length,
                         times=sample_step)
        gts.append(gt)
    return gts


def get_phase_loss(pred_phase, gt_phase):
    # pred_phase: numpy, frames x 7
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred_phase, gt_phase.squeeze())
    return loss


def get_instrument_loss(pred_instrument, gt_instrument):
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(pred_instrument, gt_instrument)
    return loss


def get_action_loss(pred_action, gt_action):
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(pred_action, gt_action)
    return loss


def num2name(num_list, feature_type):
    name_list = []
    for item in num_list:
        s = 'Hei-Chole' + str(item)
        s = s + '-' + feature_type + '.npz'
        name_list.append(s)
    return name_list


def name2num(name_list, idx):
    name = name_list[idx]
    tmp = name.split('-')
    tmp = tmp[1]
    return int(tmp[5:])


def store_info(result_dict, name_list, data_list, step, logger=None):
    l = len(name_list)
    if len(data_list) != l:
        print("store error!")
        exit()
    for i in range(l):
        result_dict[name_list[i]].append(data_list[i])
        if logger:
            logger.scalar_summary(name_list[i], data_list[i], step)


def draw_phase(pred_phase, gt_phase, naming):
    frames = pred_phase.shape[0]
    frames = np.arange(0, frames)
    pred_phase = pred_phase.cpu().numpy()
    gt_phase = gt_phase.cpu().numpy()
    plt.figure()
    plt.plot(frames, pred_phase, c='r')
    plt.plot(frames, gt_phase, c='g')
    plt.savefig(naming + '.jpg')
    plt.close()


def get_phase_acc(pred_phase, gt_phase):
    pred = pred_phase.argmax(dim=1)
    # ipdb.set_trace()
    # print(pred)
    frames = gt_phase.shape[0]
    correct_pred = 0.0
    for i in range(frames):
        if pred[i] == gt_phase[i].item():
            correct_pred += 1
    return correct_pred / frames


def kl_divergence(first_frame, second_frame):
    former_frame = F.softmax(first_frame, dim=-1)
    _kl = former_frame * (F.log_softmax(first_frame, dim=-1) -
                          F.log_softmax(second_frame, dim=-1))
    return torch.sum(_kl, dim=0)


# model output: batch_size x time_step(512*16) x class
# example based measures
def get_multi_acc(pred_multi, gt_multi):
    # ipdb.set_trace()
    c = pred_multi.shape[1]
    frames = pred_multi.shape[0]
    sig_out = sig_f(pred_multi)
    pred = torch.gt(sig_out, multi_val)
    pred = pred.cuda().float()
    eq = (pred == gt_multi)
    eq = torch.sum(eq, dim=1)
    acc = torch.sum(eq == c)
    return acc / frames


# label based measures
def get_class_prec_rec_ji_acc2(pred_multi, gt_multi, c):
    frames = pred_multi.shape[0]
    sig_out = sig_f(pred_multi)
    pred = torch.gt(sig_out, multi_val)
    pred = pred.cuda().float()
    common_c = 0.0
    pred_c = 0.0
    gt_c = 0.0
    prec = 0.0
    rec = 0.0
    ji = 0.0
    for i in range(frames):
        if gt_multi[i][c] == 1:
            gt_c += 1
            if pred[i][c] == 1:
                common_c += 1
        if pred[i][c] == 1:
            pred_c += 1
    if pred_c > 0:
        prec = common_c / pred_c
    if gt_c > 0:
        rec = common_c / gt_c
    if pred_c + gt_c - common_c > 0:
        ji = common_c / (pred_c + gt_c - common_c)
    acc2 = common_c / frames
    return prec, rec, ji, acc2


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


# lr_origin = 0.1
# print(type(varname(lr_origin)))


def board_info(str_list, info_list, step, result_dict, logger, naming):
    if naming == 1:
        for i in range(len(str_list)):
            name_list = str_list[i].split('_')
            name_list[1] = 'eval'
            str_list[i] = '_'.join(name_list)

    for i in range(len(str_list)):
        result_dict[str_list[i]].append(info_list[i])
        name_list = str_list[i].split('_')
        if logger is None:
            continue
        if name_list[1] == 'train':
            if name_list[0] == 'loss':
                logger.scalar_summary('loss/train_' + name_list[2],
                                      info_list[i], step)
        else:
            logger.scalar_summary(name_list[0] + '/' + name_list[2],
                                  info_list[i], step)



