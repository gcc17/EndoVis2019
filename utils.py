import os
import torch
import numpy as np
import random
import torch.nn as nn
from config import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
import inspect, re
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from matplotlib import gridspec
import ipdb

sig_f = nn.Sigmoid()

"""
name principle: acc/loss + train/eval/test + phase/action/instrument
"""


def get_phase_loss(pred_phase, gt_phase):
    # pred_phase: numpy, frames x 7
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred_phase, gt_phase.squeeze())
    return loss


def get_instrument_loss(pred_instrument, gt_instrument, instrument_weight):
    criterion = nn.BCEWithLogitsLoss(pos_weight=instrument_weight)
    loss = criterion(pred_instrument, gt_instrument)
    return loss


def get_action_loss(pred_action, gt_action, action_weight):
    criterion = nn.BCEWithLogitsLoss(pos_weight=action_weight)
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


def board_info(str_list, info_list, step, result_dict, writer, naming):
    if naming == 1:
        for i in range(len(str_list)):
            name_list = str_list[i].split('_')
            name_list[1] = 'eval'
            str_list[i] = '_'.join(name_list)

    for i in range(len(str_list)):
        result_dict[str_list[i]].append(info_list[i].item())
        name_list = str_list[i].split('_')
        if writer is None:
            continue
        if name_list[1] == 'train':
            if name_list[0] == 'loss':
                writer.add_scalar('loss/train_' + name_list[2], info_list[
                    i].item(), step)
        else:
            writer.add_scalar(name_list[0] + '/' + name_list[2],
                              info_list[i].item(), step)


def get_pred(feature, sample_step, instrument_category, frame_num, gt_len,
             model):
    feature_len = feature.shape[1]
    all_pred_phase = torch.zeros((feature_len * sample_step, 7)).cuda().float()
    divi_phase = torch.zeros((feature_len * sample_step, 7)).cuda().float()
    all_pred_instrument = torch.zeros(
        (feature_len * sample_step, instrument_category)).cuda().float()
    divi_instrument = torch.zeros(
        (feature_len * sample_step, instrument_category)).cuda().float()
    all_pred_action = torch.zeros((feature_len * sample_step, 4)).cuda().float()
    divi_action = torch.zeros((feature_len * sample_step, 4)).cuda().float()

    for feature_start in range(0, feature_len - clip_len + 1):
        feature_end = feature_start + clip_len
        feature_clip = feature[:, feature_start:feature_end, :]
        assert (feature_clip.shape[1] == clip_len)

        pred_phase, pred_instrument, pred_action = model(feature_clip)
        pred_phase = pred_phase.mean(0)
        pred_instrument = pred_instrument.mean(0)
        pred_action = pred_action.mean(0)

        pred_start = feature_start * sample_step
        pred_end = feature_end * sample_step
        all_pred_phase[pred_start:pred_end, :] += pred_phase
        all_pred_instrument[pred_start:pred_end, :] += pred_instrument
        all_pred_action[pred_start:pred_end, :] += pred_action
        divi_phase[pred_start:pred_end, :] += 1
        divi_instrument[pred_start:pred_end, :] += 1
        divi_action[pred_start:pred_end, :] += 1
    # overlap on test clip, use average
    all_pred_phase /= divi_phase
    all_pred_instrument /= divi_instrument
    all_pred_action /= divi_action
    # pad the prediction
    all_pred_phase = F.pad(
        all_pred_phase.permute(1, 0).unsqueeze(0),
        (frame_num + 1 - all_pred_phase.shape[0], 0),
        mode='replicate'
    ).squeeze(0).permute(1, 0)
    all_pred_instrument = F.pad(
        all_pred_instrument.permute(1, 0).unsqueeze(0),
        (frame_num + 1 - all_pred_instrument.shape[0], 0),
        mode='replicate'
    ).squeeze(0).permute(1, 0)
    all_pred_action = F.pad(
        all_pred_action.permute(1, 0).unsqueeze(0),
        (frame_num + 1 - all_pred_action.shape[0], 0),
        mode='replicate'
    ).squeeze(0).permute(1, 0)

    if all_pred_phase.shape[0] != gt_len:
        all_pred_phase = F.interpolate(
            all_pred_phase.permute(1, 0).unsqueeze(0),
            size=gt_len,
            mode='nearest'
        ).squeeze(0).permute(1, 0)
        all_pred_instrument = F.interpolate(
            all_pred_instrument.permute(1, 0).unsqueeze(0),
            size=gt_len,
            mode='nearest'
        ).squeeze(0).permute(1, 0)
        all_pred_action = F.interpolate(
            all_pred_action.permute(1, 0).unsqueeze(0),
            size=gt_len,
            mode='nearest'
        ).squeeze(0).permute(1, 0)

    return all_pred_phase, all_pred_instrument, all_pred_action


def board_pr_histo(all_gt_instrument, all_pred_instrument,
                   all_gt_action, all_pred_action,
                   log_dir, step, video_name, instrument_category):
    for c in range(instrument_category):
        pred = all_pred_instrument[:, c]
        gt = all_gt_instrument[:, c]
        instrument_log_dir = os.path.join(log_dir, 'instrument_'+str(c))
        if not os.path.exists(instrument_log_dir):
            os.makedirs(instrument_log_dir)
        instrument_writer = SummaryWriter(instrument_log_dir)
        instrument_writer.add_pr_curve(
            "Pr-Re/Instrument_" + video_name,
            gt, sig_f(pred), step)
        pred1 = []
        pred0 = []
        for i in range(all_gt_instrument.shape[0]):
            if gt[i].item() > 0:
                pred1.append(pred[i].item())
            else:
                pred0.append(pred[i].item())
        if len(pred1) > 0:
            instrument_writer.add_histogram(
                "Res/True_Instrument_" + str(c) + '_' + video_name,
                torch.Tensor(pred1).cuda().float(),
                step)
        if len(pred0) > 0:
            instrument_writer.add_histogram(
                "Res/False_Instrument_" + str(c) + '_' + video_name,
                torch.Tensor(pred0).cuda().float(),
                step)
        instrument_writer.close()

    for c in range(4):
        pred = all_pred_action[:, c]
        gt = all_gt_action[:, c]
        action_log_dir = os.path.join(log_dir, 'action_'+str(c))
        if not os.path.exists(action_log_dir):
            os.makedirs(action_log_dir)
        action_writer = SummaryWriter(action_log_dir)
        action_writer.add_pr_curve("Pr-Re/Action_" + video_name,
                            gt, sig_f(pred), step)
        pred1 = []
        pred0 = []
        for i in range(all_gt_action.shape[0]):
            if gt[i].item() > 0:
                pred1.append(pred[i].item())
            else:
                pred0.append(pred[i].item())
        if len(pred1) > 0:
            action_writer.add_histogram(
                "Res/True_Action_" + str(c) + '_' + video_name,
                torch.Tensor(pred1).cuda().float(),
                step)
        if len(pred0) > 0:
            action_writer.add_histogram(
                "Res/False_Action_" + str(c) + '_' + video_name,
                torch.Tensor(pred0).cuda().float(),
                step)






def visualize_scores_barcodes(
    titles, scores, types, class_nums=None, out_file=None, show=False):

    lens = [i.shape[0] for i in scores]
    assert (len(set(lens)) == 1)

    subplot_sum = len(titles)
    fig = plt.figure(figsize=(20, subplot_sum * 2))
    height_ratios = [1 for _ in range(subplot_sum)]
    gs = gridspec.GridSpec(subplot_sum, 1, height_ratios=height_ratios)

    for j in range(len(titles)):

        fig.add_subplot(gs[j])

        plt.xticks([])
        plt.yticks([])

        plt.title(titles[j], position=(-0.1, 0))

        axes = plt.gca()

        if types[j] == 'gt':
            barprops = dict(aspect='auto',
                            cmap=plt.cm.PiYG,
                            interpolation='nearest',
                            vmin=-1,
                            vmax=1)
        elif types[j] == 'pred':
            barprops = dict(aspect='auto',
                            cmap=plt.cm.Blues,
                            interpolation='nearest',
                            vmin=0,
                            vmax=1)
        elif types[j] == 'color':
            assert(class_nums[j] is not None)
            barprops = dict(aspect='auto',
                            cmap=plt.cm.tab10,
                            interpolation='nearest',
                            vmin=0,
                            vmax=class_nums[j]-1)
        else:
            raise Exception('Unknown Type.')

        axes.imshow(scores[j].reshape((1, -1)), **barprops)

    if out_file:
        plt.savefig(out_file)

    if show:
        plt.show()

    plt.close()




def pki(p_probs, transition_matrix, confidence_thre = 0.8, transform_thre = 50, implicit_phase = 0):
    '''
    :param p_probs: numpy array of T x C. C is num of the phase.
    :param transition_matrix: numpy array of C x C. transition_matrix[i][j] belongs in [0,1] is the transform prob from phase i to phase j.
    :param confidence_thre: hyper-parameter, the prediction confidence which under the confidence_thre will be ignored.
    :param transform_thre: hyper-parameter, the implicit phase will be changed to i after phase i continuous emergence for 'transform_thre' times.
    :param implicit_phase: default is 0.
    :return: refined labels of shape T x 1.
    '''
    t, c = p_probs.shape
    refined_labels = np.zeros(t)
    p_labels = np.argmax(p_probs, axis=1)
    p_max_probs = np.max(p_probs, axis=1)

    transform_dict = {i:0 for i in range(c)}

    for i in range(t):
        if p_max_probs[i] < confidence_thre:
            refined_labels[i] = implicit_phase
        else:
            p_label = p_labels[i]
            if p_label == implicit_phase:
                refined_labels[i] = implicit_phase
            else:
                gain = transition_matrix[implicit_phase][p_label]
                transform_dict[p_label] += gain

                #
                for k, v in transform_dict.items():
                    if v >= transform_thre:
                        print('implicit_phase changed from {} to {}'.format(implicit_phase, k))
                        implicit_phase = k
                        transform_dict = {i:0 for i in range(c)}
                        break
                refined_labels[i] = implicit_phase
    return refined_labels




def divide_avoid_zero(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def get_statistics(datadict):

    v_num = len(datadict['all_gts'])

    a_num = 4
    i_num = 7
    p_num = 7

    all_dict = {
        'action_phase': np.zeros((v_num, a_num, p_num)),  # P(phase|action) when actions happen, the prbos of phases
        'phase_action': np.zeros((v_num, p_num, a_num)),
        'action_instrument': np.zeros((v_num, a_num, i_num)),
        'instrument_action': np.zeros((v_num, i_num, a_num)),
        'instrument_phase': np.zeros((v_num, i_num, p_num)),
        'phase_instrument': np.zeros((v_num, p_num, i_num)),

        'phase_phase_0diag': np.zeros((v_num, p_num, p_num)),
        'phase_phase_non0diag': np.zeros((v_num, p_num, p_num)),

    }

    for v_id in range(v_num):

        gts_action = datadict['all_gts'][v_id]['action']
        gts_instrument = datadict['all_gts'][v_id]['instrument']
        gts_phase_raw = datadict['all_gts'][v_id]['phase']
        gts_phase = np.eye(7)[gts_phase_raw.reshape(-1).astype(int)]

        for a_id in range(a_num):

            probs = (np.tile(gts_action[:,a_id:a_id+1], (1,i_num)) * gts_instrument).sum(0)
            if gts_action[:,a_id:a_id+1].sum() != 0:
                probs = probs / gts_action[:,a_id:a_id+1].sum()
            all_dict['action_instrument'][v_id, a_id, :] = probs

            probs = (np.tile(gts_action[:,a_id:a_id+1], (1,p_num)) * gts_phase).sum(0)
            if gts_action[:,a_id:a_id+1].sum() != 0:
                probs = probs / gts_action[:,a_id:a_id+1].sum()
            all_dict['action_phase'][v_id, a_id, :] = probs

        for i_id in range(i_num):

            probs = (np.tile(gts_instrument[:,i_id:i_id+1], (1,a_num)) * gts_action).sum(0)
            if gts_instrument[:,i_id:i_id+1].sum() != 0:
                probs = probs / gts_instrument[:,i_id:i_id+1].sum()
            all_dict['instrument_action'][v_id, i_id, :] = probs

            probs = (np.tile(gts_instrument[:,i_id:i_id+1], (1,p_num)) * gts_phase).sum(0)
            if gts_instrument[:,i_id:i_id+1].sum() != 0:
                probs = probs / gts_instrument[:,i_id:i_id+1].sum()
            all_dict['instrument_phase'][v_id, i_id, :] = probs

        for p_id in range(p_num):

            probs = (np.tile(gts_phase[:,p_id:p_id+1], (1,i_num)) * gts_instrument).sum(0)
            if gts_phase[:,p_id:p_id+1].sum() != 0:
                probs = probs / gts_phase[:,p_id:p_id+1].sum()
            all_dict['phase_instrument'][v_id, p_id, :] = probs

            probs = (np.tile(gts_phase[:,p_id:p_id+1], (1,a_num)) * gts_action).sum(0)
            if gts_phase[:,p_id:p_id+1].sum() != 0:
                probs = probs / gts_phase[:,p_id:p_id+1].sum()
            all_dict['phase_action'][v_id, p_id, :] = probs


        for i in range(gts_phase_raw.shape[0]-1):
            if gts_phase_raw[i,0] != gts_phase_raw[i+1,0]:
                all_dict['phase_phase_0diag'][v_id, int(gts_phase_raw[i,0]), int(gts_phase_raw[i+1,0])] += 1
            all_dict['phase_phase_non0diag'][v_id, int(gts_phase_raw[i,0]), int(gts_phase_raw[i+1,0])] += 1


        all_dict['phase_phase_0diag'][v_id] = divide_avoid_zero(
            all_dict['phase_phase_0diag'][v_id].T,
            all_dict['phase_phase_0diag'][v_id].sum(1)).T

        all_dict['phase_phase_non0diag'][v_id] = divide_avoid_zero(
            all_dict['phase_phase_non0diag'][v_id].T,
            all_dict['phase_phase_non0diag'][v_id].sum(1)).T    # TO FIX PHASE -2, NOT SHOW IN ALL VIDEOS


    keys = [i for i in all_dict.keys()]
    for k in keys:
        all_dict[k + '_mean'] = all_dict[k].mean(0)
        all_dict[k + '_std'] = all_dict[k].std(0)

        assert(all_dict[k + '_mean'].max() <= 1)
        assert(all_dict[k + '_mean'].min() >= 0)
        assert(all_dict[k + '_std'].max() <= 1)
        assert(all_dict[k + '_std'].min() >= 0)

        # print(k)

        # plt.matshow(all_dict[k + '_mean'])
        # plt.show()

        # plt.matshow(all_dict[k + '_std'])
        # plt.show()

        # print(all_dict[k + '_std'].max())

    print(all_dict.keys())

    return all_dict


def get_instrument_action_correlation(pred_instrument, pred_action,
                                      instrument_action_mean, instrument_action_std,
                                      action_instrument_mean, action_instrument_std,
                                      mode=0):
    action_instrument_mean = torch.Tensor(action_instrument_mean).cuda().float()
    instrument_action_mean = torch.Tensor(instrument_action_mean).cuda().float()
    # ipdb.set_trace()

    if mode == 0:
        instrument_action_std = torch.Tensor(instrument_action_std.mean(0)).cuda().float()
        action_instrument_std = torch.Tensor(action_instrument_std.mean(0)).cuda().float()
        instrument_weight = torch.ones(action_instrument_std.shape).cuda().float() \
                            - action_instrument_std
        action_weight = torch.ones(instrument_action_std.shape).cuda().float()\
                        - instrument_action_std
        instrument_criterion = nn.BCEWithLogitsLoss(weight=instrument_weight)
        action_criterion = nn.BCEWithLogitsLoss(weight=action_weight)

        infer_action = torch.mm(pred_instrument, instrument_action_mean)
        infer_instrument = torch.mm(pred_action, action_instrument_mean)

        bce_instrument = instrument_criterion(pred_instrument,
                                              sig_f(infer_instrument))
        bce_action = action_criterion(pred_action, sig_f(infer_action))
        return bce_instrument + bce_action

    else:
        action_instrument_std = torch.Tensor(action_instrument_std).cuda().float()
        instrument_action_std = torch.Tensor(instrument_action_std).cuda().float()

        action_instrument_rand = torch.normal(action_instrument_mean,
                                              action_instrument_std)
        instrument_action_rand = torch.normal(instrument_action_mean,
                                              instrument_action_std)

        infer_action = torch.mm(pred_instrument, instrument_action_rand)
        infer_instrument = torch.mm(pred_action, action_instrument_rand)
        criterion = nn.BCEWithLogitsLoss()
        bce_instrument = criterion(pred_instrument, sig_f(infer_instrument))
        bce_action = criterion(pred_action, sig_f(infer_action))
        return bce_instrument+bce_action


def get_phase_correlation(pred_phase, phase_phase_non0diag_mean,
                          phase_phase_non0diag_std, mode=0):
    phase_phase_non0diag_mean = torch.Tensor(phase_phase_non0diag_mean).cuda().float()

    if mode == 0:
        phase_phase_non0diag_std = torch.Tensor(phase_phase_non0diag_std.mean(0)).cuda().float()
        phase_weight = torch.ones(phase_phase_non0diag_std.shape).cuda().float()\
                       - phase_phase_non0diag_std
        phase_criterion = nn.BCEWithLogitsLoss(weight=phase_weight)
        infer_phase = torch.mm(pred_phase, phase_phase_non0diag_mean)
        bce_phase = phase_criterion(pred_phase, sig_f(infer_phase))
        return bce_phase

    else:
        phase_phase_non0diag_std = torch.Tensor(phase_phase_non0diag_std).cuda().float()
        phase_phase_rand = torch.normal(phase_phase_non0diag_mean,
                                        phase_phase_non0diag_std)
        infer_phase = torch.mm(pred_phase, phase_phase_rand)
        criterion = nn.BCEWithLogitsLoss()
        bce_phase = criterion(pred_phase, sig_f(infer_phase))
        return bce_phase


instrument_category = 7
def get_pos_weight(gt_root_dir, train_list):
    """
    :param gt_root_dir: annotation root directory
    :param train_list: video number in training dataset, eg.1,2,3,4
    :return: instrument pos_weight, action pos_weight, torch.Tensor
    """
    instrument_dir = os.path.join(gt_root_dir, 'Instrument')
    instrument_pos_weight = np.zeros(instrument_category)
    for num in train_list:
        gt_file = os.path.join(instrument_dir, "hei-chole"+str(num)+"_annotation_instrument.csv")
        gt_data = np.loadtxt(gt_file, delimiter=',')
        gt_data = gt_data[:, 1:instrument_category+1]
        total_frames = gt_data.shape[0]
        pos_label = gt_data.sum(0)
        for i in range(instrument_category):
            if pos_label[i] == 0:
                print("video", str(num), "instrument", str(i))
                pos_label[i] = 1000
            instrument_pos_weight[i] += float(total_frames-pos_label[i])/pos_label[i]
    instrument_pos_weight /= len(train_list)

    action_dir = os.path.join(gt_root_dir, 'Action')
    action_pos_weight = np.zeros(4)
    for num in train_list:
        gt_file = os.path.join(action_dir, "hei-chole"+str(num)+"_annotation_action.csv")
        gt_data = np.loadtxt(gt_file, delimiter=',')
        gt_data = gt_data[:, 1:]
        total_frames = gt_data.shape[0]
        pos_label = gt_data.sum(0)
        for i in range(4):
            if pos_label[i] == 0:
                print("video", str(num), "action", str(i))
                pos_label[i] = 200
            action_pos_weight[i] += float(total_frames-pos_label[i])/pos_label[i]
    action_pos_weight /= len(train_list)
    return torch.Tensor(instrument_pos_weight), \
           torch.Tensor(action_pos_weight)


# video_list = []
# for i in range(1, 25):
#     video_list.append(i)
# print(get_pos_weight("../New_Annotations", video_list))


def get_concurrence(gt_root_dir, train_list):
    """
    :param gt_root_dir: annotation root directory
    :param train_list: video number in training dataset, eg.1,2,3,4
    :return: instrument and action concurrence matrix, torch.Tensor
    matrix[i,j]: when there is instrument i, the prob of instrument j
    """
    instrument_dir = os.path.join(gt_root_dir, 'Instrument')
    instrument_concurrence = np.zeros([instrument_category, instrument_category])
    for num in train_list:
        video_instrument = np.zeros([instrument_category, instrument_category])
        gt_file = os.path.join(instrument_dir, "hei-chole"+str(num)+"_annotation_instrument.csv")
        gt_data = np.loadtxt(gt_file, delimiter=',')
        gt_data = gt_data[:, 1:instrument_category+1]
        total_frames = gt_data.shape[0]
        for frame in range(total_frames):
            frame_instrument = gt_data[frame].sum()
            if frame_instrument == 1:
                single_instrument = gt_data[frame].argmax()
                video_instrument[single_instrument, single_instrument] += 1
            elif frame_instrument > 1:
                for i in range(instrument_category-1):
                    if gt_data[frame, i] > 0:
                        for j in range(i+1, instrument_category):
                            if gt_data[frame, j] > 0:
                                video_instrument[i, j] += 1
                                video_instrument[j, i] += 1
        for i in range(instrument_category):
            instrument_sum = float(video_instrument[i].sum())
            if instrument_sum > 0:
                video_instrument[i] /= instrument_sum
        instrument_concurrence += video_instrument
    instrument_concurrence /= len(train_list)

    action_dir = os.path.join(gt_root_dir, 'Action')
    action_concurrence = np.zeros([4, 4])
    for num in train_list:
        video_action = np.zeros([4, 4])
        gt_file = os.path.join(action_dir, "hei-chole"+str(num)+"_annotation_action.csv")
        gt_data = np.loadtxt(gt_file, delimiter=',')
        gt_data = gt_data[:, 1:]
        total_frames = gt_data.shape[0]
        for frame in range(total_frames):
            frame_action = gt_data[frame].sum()
            if frame_action == 1:
                single_action = gt_data[frame].argmax()
                video_action[single_action, single_action] += 1
            elif frame_action > 1:
                for i in range(3):
                    if gt_data[frame, i] > 0:
                        for j in range(i+1, 4):
                            if gt_data[frame, j] > 0:
                                video_action[i, j] += 1
                                video_action[j, i] += 1
        for i in range(4):
            action_sum = float(video_action[i].sum())
            if action_sum > 0:
                video_action[i] /= action_sum
        action_concurrence += video_action
    action_concurrence /= len(train_list)

    return instrument_concurrence, action_concurrence

# video_list = []
# for i in range(1, 25):
#     video_list.append(i)
# print(get_concurrence("../New_Annotations", video_list))

"""(array([[2.50104383e-01, 4.16514394e-02, 4.99913807e-01, 2.40620846e-02,
        6.13024397e-02, 1.22728053e-01, 2.37793278e-04],
       [6.86386976e-01, 2.94373867e-01, 0.00000000e+00, 0.00000000e+00,
        1.21326085e-02, 7.10654876e-03, 0.00000000e+00],
       [6.90699566e-01, 0.00000000e+00, 2.84505086e-01, 0.00000000e+00,
        1.33046762e-02, 1.14906715e-02, 0.00000000e+00],
       [6.14625420e-01, 0.00000000e+00, 0.00000000e+00, 3.29401786e-01,
        1.43061271e-02, 0.00000000e+00, 0.00000000e+00],
       [3.35113481e-01, 2.79282522e-03, 2.46521477e-02, 8.52125718e-03,
        8.02549448e-02, 1.31998677e-01, 0.00000000e+00],
       [6.87648434e-01, 1.69395308e-03, 3.37768889e-02, 0.00000000e+00,
        1.02093835e-01, 1.74786889e-01, 0.00000000e+00],
       [1.04864120e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 3.11802547e-02]]), 
 array([[2.72988415e-01, 7.26047514e-01, 0.00000000e+00, 9.64070730e-04],
       [1.15734004e-02, 9.78807234e-01, 4.04548840e-03, 5.57387713e-03],
       [0.00000000e+00, 9.37621378e-01, 2.07119553e-02, 0.00000000e+00],
       [3.01293357e-03, 9.73787284e-01, 0.00000000e+00, 2.31997829e-02]]))
"""
"""instrument_concurrence = \
[[0.25, 0.04, 0.50, 0.02, 0.06, 0.12, 0], 
 [0.69, 0.29, 0, 0, 0.01, 0.01, 0], 
 [0.69, 0, 0.28, 0, 0.01, 0.01, 0], 
 [0.61, 0, 0, 0.33, 0.01, 0, 0], 
 [0.34, 0.01, 0.02, 0.01, 0.08, 0.13, 0], 
 [0.69, 0, 0.02, 0.01, 0.08, 0.13, 0], 
 [0.01, 0, 0, 0, 0, 0, 0.03]]
action_concurrence = \
[[0.27, 0.73, 0, 0], 
 [0.01, 0.98, 0, 0.01], 
 [0, 0.94, 0.02, 0], 
 [0, 0.97, 0, 0.02]]
"""