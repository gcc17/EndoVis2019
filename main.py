import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
from new_dataset import TrainDataset, TestDataset
from model import TCNNet, GRUNet
import os
from config import *
from utils import get_phase_loss, get_instrument_loss, get_action_loss, \
    num2name, get_phase_acc, get_multi_acc, \
    get_class_prec_rec_ji_acc2, board_info
import ipdb
from torch.utils.tensorboard import SummaryWriter
from new_dataset import *

"""
name principle: acc/loss + train/eval/test + phase/action/instrument
"""

######################

locals().update(training_params)
instrument_category = 7


######################


def test(model, test_loader, sample_step, step, result_dict, test_logger,
         logger, naming):
    """
    :param model: 
    :param test_loader: 
    :param sample_step:
    :param step: 
    :param result_dict: 
    :param test_logger: used for scalar logging
    :param logger: used for image and precision recall curve logging
    :param naming: 
    :return: 
    
    data['feature'].shape                   # (1, 3433, 1024)
    data['action'].shape                    # (54930, 4)
    data['action_detailed']                 # (54930, 12)
    data['instrument'].shape                # (54930, 21)
    data['instrument_detailed'].shape       # (54930, 31)
    data['phase'].shape                     # (54930, 1)
    data['video_name']                      # Hei-Chole1 (For Debug)
    data['frame_num']                       # 54929 (For Debug)
    data['gt_len']                          # 54930 (For Debug)

    """
    video_num = 0
    loss_test_phase = 0.0
    loss_test_instrument = 0.0
    loss_test_action = 0.0
    acc_test_phase = 0.0
    acc_test_instrument = 0.0
    acc_test_action = 0.0
    prec_test_instrument = np.zeros(instrument_category)
    prec_test_action = np.zeros(4)
    rec_test_instrument = np.zeros(instrument_category)
    rec_test_action = np.zeros(4)
    ji_test_instrument = np.zeros(instrument_category)
    ji_test_action = np.zeros(4)
    acc2_test_instrument = np.zeros(instrument_category)
    acc2_test_action = np.zeros(4)

    with torch.no_grad():
        for num, data in enumerate(test_loader):
            video_num += 1
            feature = data['feature'].squeeze(0).cuda().float()
            # 1/10 x len x feature_dim
            all_gt_phase = data['phase'].squeeze(0).cuda().long()
            all_gt_instrument = data['instrument'].squeeze(0).cuda().float()
            all_gt_action = data['action'].squeeze(0).cuda().float()
            gt_len = data['gt_len'].item()
            frame_num = data['frame_num'].item()
            video_name = data['video_name'][0]

            print('Testing {}'.format(video_name))

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

            assert (all_pred_action.shape == all_gt_action.shape)

            loss_test_phase += get_phase_loss(all_pred_phase, all_gt_phase)
            loss_test_instrument += get_instrument_loss(
                all_pred_instrument, all_gt_instrument)
            loss_test_action += get_action_loss(all_pred_action, all_gt_action)

            acc_test_phase += get_phase_acc(all_pred_phase, all_gt_phase)
            acc_test_instrument += get_multi_acc(all_pred_instrument,
                                                 all_gt_instrument)
            acc_test_action += get_multi_acc(all_pred_action, all_gt_action)

            for c in range(instrument_category):
                prec0, rec0, ji0, acc20 = get_class_prec_rec_ji_acc2(
                    all_pred_instrument, all_gt_instrument, c)
                prec_test_instrument[c] += prec0
                rec_test_instrument[c] += rec0
                ji_test_instrument[c] += ji0
                acc2_test_instrument[c] += acc20
            for c in range(4):
                prec0, rec0, ji0, acc20 = get_class_prec_rec_ji_acc2(
                    all_pred_action, all_gt_action, c)
                prec_test_action[c] += prec0
                rec_test_action[c] += rec0
                ji_test_action[c] += ji0
                acc2_test_action[c] += acc20

    loss_test_phase /= video_num
    loss_test_instrument /= video_num
    loss_test_action /= video_num
    acc_test_phase /= video_num
    acc_test_instrument /= video_num
    acc_test_action /= video_num
    for c in range(instrument_category):
        prec_test_instrument[c] /= video_num
        rec_test_instrument[c] /= video_num
        ji_test_instrument[c] /= video_num
        acc2_test_instrument[c] /= video_num
    for c in range(4):
        prec_test_action[c] /= video_num
        rec_test_action[c] /= video_num
        ji_test_action[c] /= video_num
        acc2_test_action[c] /= video_num

    mprec_test_instrument = prec_test_instrument.sum() / instrument_category
    mrec_test_instrument = rec_test_instrument.sum() / instrument_category
    mji_test_instrument = ji_test_instrument.sum() / instrument_category
    macc2_test_instrument = acc2_test_instrument.sum() / instrument_category
    mprec_test_action = prec_test_action.sum() / 4
    mrec_test_action = rec_test_action.sum() / 4
    mji_test_action = ji_test_action.sum() / 4
    macc2_test_action = acc2_test_action.sum() / 4

    str_list = ['loss_test_phase', 'loss_test_instrument', 'loss_test_action',
                'acc_test_phase', 'acc_test_instrument', 'acc_test_action',
                'mprec_test_instrument', 'mprec_test_action',
                'mrec_test_instrument', 'mrec_test_action',
                'mji_test_instrument', 'mji_test_action',
                'macc2_test_instrument', 'macc2_test_action']
    info_list = []
    for i in range(len(str_list)):
        info_list.append(eval(str_list[i]))
    board_info(str_list, info_list, step, result_dict, test_logger, naming)


def train(model, train_loader, eval_loader, test_loader, naming, use_tf_log):
    model_dir = os.path.join('./models', naming)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log_dir = os.path.join('./logs', naming)
    train_log_dir = os.path.join(log_dir, 'summary/train')
    eval_log_dir = os.path.join(log_dir, 'summary/eval')
    test_log_dir = os.path.join(log_dir, 'summary/test')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.makedirs(train_log_dir)
        os.makedirs(eval_log_dir)
        os.makedirs(test_log_dir)
    if use_tf_log:
        from logger import Logger
        train_logger = Logger(train_log_dir)
        eval_logger = Logger(eval_log_dir)
        test_logger = Logger(test_log_dir)
        file_logger = SummaryWriter(log_dir)
    else:
        train_logger = None
        eval_logger = None
        test_logger = None
        file_logger = None

    result_dict = {'loss_train_phase': [], 'loss_train_instrument': [],
                   'loss_train_action': [],
                   'loss_eval_phase': [], 'loss_eval_instrument': [],
                   'loss_eval_action': [],
                   'loss_test_phase': [], 'loss_test_instrument': [],
                   'loss_test_action': [],
                   'acc_eval_phase': [], 'acc_eval_instrument': [],
                   'acc_eval_action': [],
                   'acc_test_phase': [], 'acc_test_instrument': [],
                   'acc_test_action': [],
                   'mprec_eval_instrument': [], 'mprec_eval_action': [],
                   'mprec_test_instrument': [], 'mprec_test_action': [],
                   'mrec_eval_instrument': [], 'mrec_eval_action': [],
                   'mrec_test_instrument': [], 'mrec_test_action': [],
                   'mji_eval_instrument': [], 'mji_eval_action': [],
                   'mji_test_instrument': [], 'mji_test_action': [],
                   'macc2_eval_instrument': [], 'macc2_eval_action': [],
                   'macc2_test_instrument': [], 'macc2_test_action':[]}

    # loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=weight_decay)

    optimizer.zero_grad()

    phase_criterion = nn.CrossEntropyLoss()
    instrument_criterion = nn.BCEWithLogitsLoss()
    action_criterion = nn.BCEWithLogitsLoss()

    step = 0
    # ipdb.set_trace()

    # Train loop
    while step < max_step_num:
        for _, data in enumerate(train_loader):

            if step % log_freq == 0:
                print("begin test...")
                model.eval()
                test(model, test_loader, sample_step, step, result_dict,
                     test_logger, file_logger, naming)
                torch.save(model.state_dict(),
                           os.path.join(model_dir, 'model'))
                test(model, eval_loader, sample_step, step, result_dict,
                     eval_logger, file_logger, 1)

            model.train()
            gt_phase = data['phase'].cuda().long()
            gt_instrument = data['instrument'].cuda().float()
            gt_action = data['action'].cuda().float()
            feature = data['feature'].cuda().float()

            # ipdb.set_trace()

            pred_phase, pred_instrument, pred_action = model(feature)
            # output: batch_size x time_step x class
            loss_train_phase = 0.0
            loss_train_instrument = 0.0
            loss_train_action = 0.0
            for i in range(batch_size):
                loss_train_phase += phase_criterion(pred_phase[i],
                                                    gt_phase[i].squeeze())
                loss_train_instrument += instrument_criterion(
                    pred_instrument[i], gt_instrument[i])
                loss_train_action += action_criterion(pred_action[i],
                                                      gt_action[i])

            loss_train_phase = loss_train_phase / batch_size
            loss_train_instrument = loss_train_instrument / batch_size
            loss_train_action = loss_train_action / batch_size
            loss_train_total = loss_train_phase + loss_train_instrument + loss_train_action

            # print and save calculation
            #####################################
            print('{} -- Step {}: Loss_total-{}'.format(naming, step,
                                                        loss_train_total.item()))
            str_list = ['loss_train_phase', 'loss_train_instrument',
                        'loss_train_action']
            info_list = [loss_train_phase, loss_train_instrument,
                         loss_train_action]
            board_info(str_list, info_list, step, result_dict, train_logger, 0)
            #####################################

            optimizer.zero_grad()
            loss_train_total.backward()
            optimizer.step()

            step += 1
            if step >= max_step_num:
                break
    np.save(os.path.join(log_dir, naming + '.npy'), result_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--feature_type', type=str)
    parser.add_argument('--use_tf_log', type=int)

    args = parser.parse_args()

    feature_type = args.feature_type
    use_tf_log = args.use_tf_log

    case_splits = np.load('metas/case_splits.npy', allow_pickle=True).item()
    for repeat_id in range(split_repeat):
        for split_id in range(split_num):
            train_cases = list(case_splits[repeat_id])
            train_cases.pop(split_id)
            train_list = np.concatenate(train_cases)
            test_list = case_splits[repeat_id][split_id]
            train_name_list = num2name(train_list, feature_type)
            test_name_list = num2name(test_list, feature_type)

            naming = '{}-{}-repeat_{}-split_{}'.format(naming_prefix,
                                                       feature_type,
                                                       repeat_id, split_id)

            print('Feature_type:    {}'.format(feature_type))
            print('TFLog:           {}'.format(use_tf_log))
            print('Naming:          {}'.format(naming))

            if model_type == 'TCN':
                model = TCNNet(input_dim=input_dim,
                               dropout_rate=dropout_rate).cuda()

            elif model_type == 'GRU':
                model = GRUNet(input_dim=input_dim,
                               dropout_rate=dropout_rate).cuda()

            else:
                raise Exception('Unknown Model Type.')



            train_datadict = get_datadict(
                gt_root_dir='../New_Annotations',
                feature_dir='../i3d/rgb_oversample_4',
                feature_files=train_name_list,
                sample_step=sample_step
            )
            test_datadict = get_datadict(
                gt_root_dir='../New_Annotations',
                feature_dir='../i3d/rgb_oversample_4',
                feature_files=test_name_list,
                sample_step=sample_step
            )

            train_dataset = TrainDataset(train_datadict, clip_len, sample_step)
            test_dataset = TestDataset(test_datadict, clip_len)
            eval_dataset = TestDataset(train_datadict, clip_len)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       drop_last=True)
            eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=1,
                                                      shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=1,
                                                      shuffle=True)

            train(model, train_loader, eval_loader, test_loader, naming,
                  use_tf_log)
