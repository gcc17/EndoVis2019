import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
from model import TCNNet, GRUNet, MLPNet
import os
from config import *
from utils import get_phase_loss, get_instrument_loss, get_action_loss, \
    num2name, board_info, get_pred, board_pr_histo, \
    visualize_scores_barcodes, get_statistics, \
    get_instrument_action_correlation, get_phase_correlation, pki
from torch.utils.tensorboard import SummaryWriter
from new_dataset import *
from sklearn.metrics import average_precision_score, accuracy_score, \
    precision_recall_fscore_support
import pdb

"""
name principle: acc/loss + train/eval/test + phase/action/instrument
"""

######################

locals().update(training_params)
instrument_category = 7
test_video = []
eval_video = []
sig_f = nn.Sigmoid()
# action_weight = [ 30, 1, 60, 200 ]
# instrument_weight = [ 1, 20, 2, 10, 5, 10 ]
action_weight = [9.0, 0.33, 61.9, 29.5]
instrument_weight = [0.66, 30.19, 1.38, 65.25, 32.69, 12.74, 72.24]



######################


def test(model, test_loader, sample_step, step, all_dict, result_dict,
         writer, log_dir, naming, iseval, model2=None):
    """
    :param model: 
    :param test_loader: 
    :param sample_step:
    :param step: 
    :param result_dict: 
    :param writer: log information
    :param log_dir: new writer to log pr curve
    :param naming: 
    :param model2: model for late fusion
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
    corloss_test_action_instrument = 0.0
    corloss_test_phase = 0.0
    acc_test_phase = 0.0
    acc_test_instrument = 0.0
    acc_test_action = 0.0
    prec_test_instrument = 0.0
    prec_test_action = 0.0
    rec_test_instrument = 0.0
    rec_test_action = 0.0
    f1_test_instrument = 0.0
    f1_test_action = 0.0
    mAP_test_instrument = 0.0
    mAP_test_action = 0.0

    with torch.no_grad():
        for num, data in enumerate(test_loader):
            video_num += 1
            fusion_mode = data['fusion_mode'].item()
            if fusion_mode == 0 or fusion_mode == 1 or fusion_mode == 3 or \
                            fusion_mode == 4 or fusion_mode == 6:
                feature = data['feature'].squeeze(0).cuda().float()
            else:
                rgb_feature = data['rgb_feature'].squeeze(0).cuda().float()
                flow_feature = data['flow_feature'].squeeze(0).cuda().float()

            all_gt_phase = data['phase'].squeeze(0).cuda().long()
            all_gt_instrument = data['instrument'].squeeze(0).cuda().float()
            all_gt_action = data['action'].squeeze(0).cuda().float()
            gt_len = data['gt_len'].item()
            frame_num = data['frame_num'].item()
            video_name = data['video_name'][0]

            print('Testing {}'.format(video_name))
            if fusion_mode == 0 or fusion_mode == 1 or fusion_mode == 3 or \
                            fusion_mode == 4 or fusion_mode == 6:
                all_pred_phase, all_pred_instrument, all_pred_action = get_pred(
                    feature, sample_step, instrument_category, frame_num,
                    gt_len,
                    model)
            else:
                pred_phase1, pred_instrument1, pred_action1 = get_pred(
                    rgb_feature, sample_step, instrument_category, frame_num,
                    gt_len, model)
                pred_phase2, pred_instrument2, pred_action2 = get_pred(
                    flow_feature, sample_step, instrument_category, frame_num,
                    gt_len, model2)
                all_pred_phase = (pred_phase1 + pred_phase2) / 2
                all_pred_instrument = (pred_instrument1 + pred_instrument2) / 2
                all_pred_action = (pred_action1 + pred_action2) / 2

            assert (all_pred_action.shape == all_gt_action.shape)
            revised_pred_phase = all_pred_phase.cpu().numpy()
            revised_pred_phase = pki(revised_pred_phase, all_dict[
                'phase_phase_0diag_mean'])
            revised_pred_phase = torch.Tensor(revised_pred_phase).cuda().float()

            titles = []
            scores = []
            types = []
            for i in range(4):
                titles.append('a{}_gt'.format(i))
                scores.append(all_gt_action.cpu().numpy()[:, i])
                types.append('gt')

                titles.append('a{}_pred'.format(i))
                scores.append(sig_f(all_pred_action).cpu().numpy()[:, i])
                types.append('pred')

            image_dir = os.path.join('./images', naming)
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            visualize_scores_barcodes(titles, scores, types,
                                      class_nums=None,
                                      out_file='{}/{}_{}_{}.png'.format(
                                          image_dir, step, video_name,
                                          'action'),
                                      show=False)

            titles = []
            scores = []
            types = []
            for i in range(7):
                titles.append('i{}_gt'.format(i))
                scores.append(all_gt_instrument.cpu().numpy()[:, i])
                types.append('gt')

                titles.append('i{}_pred'.format(i))
                scores.append(sig_f(all_pred_instrument).cpu().numpy()[:, i])
                types.append('pred')

            visualize_scores_barcodes(titles, scores, types,
                                      class_nums=None,
                                      out_file='{}/{}_{}_{}.png'.format(
                                          image_dir, step, video_name,
                                          'instrument'),
                                      show=False)

            visualize_scores_barcodes(
                titles=['p_gt', 'p_pred', 'p_revised'],
                scores=[all_gt_phase.cpu().numpy(), all_pred_phase.argmax(
                    dim=1).cpu().numpy(), revised_pred_phase.cpu().numpy()],
                types=['color', 'color', 'color'],
                class_nums=[7, 7, 7],
                out_file='{}/{}_{}_{}.png'.format(image_dir, step,
                                                         video_name, 'phase'),
                show=False)

            # work around
            temp_all_gt_action = all_gt_action.cpu().numpy()
            temp_all_gt_instrument = all_gt_instrument.cpu().numpy()

            for cc in range(temp_all_gt_action.shape[1]):
                if temp_all_gt_action[:, cc].sum() == 0:
                    print('[WARNING] fix all neg gt!')
                    temp_all_gt_action[0, cc] = 1

            for cc in range(temp_all_gt_instrument.shape[1]):
                if temp_all_gt_instrument[:, cc].sum() == 0:
                    print('[WARNING] fix all neg gt!')
                    temp_all_gt_instrument[0, cc] = 1

            mAP_test_action += average_precision_score(
                temp_all_gt_action, sig_f(all_pred_action).cpu().numpy())
            mAP_test_instrument += average_precision_score(
                temp_all_gt_instrument,
                sig_f(all_pred_instrument).cpu().numpy())

            loss_test_phase += get_phase_loss(all_pred_phase, all_gt_phase)
            loss_test_instrument += get_instrument_loss(
                all_pred_instrument, all_gt_instrument, instrument_weight)
            loss_test_action += get_action_loss(all_pred_action,
                                                all_gt_action, action_weight)
            corloss_test_action_instrument += \
                get_instrument_action_correlation(all_pred_instrument,
                                                  all_pred_action,
                                                  all_dict[
                                                      'instrument_action_mean'],
                                                  all_dict[
                                                      'instrument_action_std'],
                                                  all_dict[
                                                      'action_instrument_mean'],
                                                  all_dict[
                                                      'action_instrument_std'],
                                                  mode=cor_mode)
            corloss_test_phase += \
                get_phase_correlation(all_pred_phase,
                                      all_dict['phase_phase_non0diag_mean'],
                                      all_dict['phase_phase_non0diag_std'],
                                      mode=cor_mode)

            binary_pred_instrument = torch.gt(sig_f(all_pred_instrument),
                                              multi_val).cpu().numpy()
            binary_pred_action = torch.gt(sig_f(all_pred_action),
                                          multi_val).cpu().numpy()

            acc_test_phase += accuracy_score(all_gt_phase.cpu().numpy(),
                                             revised_pred_phase.cpu().numpy())
            acc_test_instrument += accuracy_score(temp_all_gt_instrument,
                                                  binary_pred_instrument)
            acc_test_action += accuracy_score(temp_all_gt_action,
                                              binary_pred_action)

            prec_instrument, rec_instrument, f1_instrument, _ = \
                precision_recall_fscore_support(temp_all_gt_instrument,
                                                binary_pred_instrument,
                                                average='macro')
            prec_test_instrument += prec_instrument
            rec_test_instrument += rec_instrument
            f1_test_instrument += f1_instrument
            prec_action, rec_action, f1_action, _ = \
                precision_recall_fscore_support(temp_all_gt_action,
                                                binary_pred_action,
                                                average='macro')
            prec_test_action += prec_action
            rec_test_action += rec_action
            f1_test_action += f1_action

            if video_name in eval_video or video_name in test_video:
                print("drawing pr and histo")
                board_pr_histo(all_gt_instrument, all_pred_instrument,
                               all_gt_action, all_pred_action,
                               log_dir, step, video_name, instrument_category)

    loss_test_phase /= video_num
    loss_test_instrument /= video_num
    loss_test_action /= video_num
    corloss_test_action_instrument /= video_num
    corloss_test_phase /= video_num
    acc_test_phase /= video_num
    acc_test_instrument /= video_num
    acc_test_action /= video_num
    prec_test_instrument /= video_num
    prec_test_action /= video_num
    rec_test_instrument /= video_num
    rec_test_action /= video_num
    f1_test_instrument /= video_num
    f1_test_action /= video_num
    mAP_test_action /= video_num
    mAP_test_instrument /= video_num

    str_list = ['loss_test_phase', 'loss_test_instrument', 'loss_test_action',
                'acc_test_phase', 'acc_test_instrument', 'acc_test_action',
                'prec_test_instrument', 'prec_test_action',
                'rec_test_instrument', 'rec_test_action',
                'f1_test_instrument', 'f1_test_action',
                'mAP_test_instrument', 'mAP_test_action']
    info_list = []
    for i in range(len(str_list)):
        info_list.append(eval(str_list[i]))
    board_info(str_list, info_list, step, result_dict, writer, iseval)
    if iseval == 0:
        result_dict['corloss_test_action_instrument'].append(
            corloss_test_action_instrument.item())
        writer.add_scalar('loss/test_cor_action_instrument',
                          corloss_test_action_instrument.item(), step)
        result_dict['corloss_test_phase'].append(
            corloss_test_action_instrument.item())
        writer.add_scalar('loss/test_cor_phase', corloss_test_phase.item(),
                          step)
    else:
        result_dict['corloss_eval_action_instrument'].append(
            corloss_test_action_instrument.item())
        writer.add_scalar('loss/eval_cor_action_instrument',
                          corloss_test_action_instrument.item(), step)
        result_dict['corloss_eval_phase'].append(
            corloss_test_action_instrument.item())
        writer.add_scalar('loss/eval_cor_phase', corloss_test_phase.item(),
                          step)


def train(model, train_loader, eval_loader, test_loader, naming, use_tf_log,
          sample_step, all_dict, model2=None):
    print("begin training...")
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
        train_writer = SummaryWriter(train_log_dir)
        eval_writer = SummaryWriter(eval_log_dir)
        test_writer = SummaryWriter(test_log_dir)
    else:
        train_writer = None
        eval_writer = None
        test_writer = None

    result_dict = {'loss_train_phase': [], 'loss_train_instrument': [],
                   'loss_train_action': [],
                   'corloss_train_action_instrument': [],
                   'corloss_train_phase': [],
                   'loss_eval_phase': [], 'loss_eval_instrument': [],
                   'loss_eval_action': [],
                   'corloss_eval_action_instrument': [],
                   'corloss_eval_phase': [],
                   'loss_test_phase': [], 'loss_test_instrument': [],
                   'loss_test_action': [],
                   'corloss_test_action_instrument': [],
                   'corloss_test_phase': [],
                   'acc_eval_phase': [], 'acc_eval_instrument': [],
                   'acc_eval_action': [],
                   'acc_test_phase': [], 'acc_test_instrument': [],
                   'acc_test_action': [],
                   'prec_eval_instrument': [], 'prec_eval_action': [],
                   'prec_test_instrument': [], 'prec_test_action': [],
                   'rec_eval_instrument': [], 'rec_eval_action': [],
                   'rec_test_instrument': [], 'rec_test_action': [],
                   'f1_eval_instrument': [], 'f1_eval_action': [],
                   'f1_test_instrument': [], 'f1_test_action': [],
                   'mAP_eval_instrument': [], 'mAP_eval_action': [],
                   'mAP_test_instrument': [], 'mAP_test_action': []}

    # loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=weight_decay)

    optimizer.zero_grad()

    print(action_weight)
    print(instrument_weight)

    phase_criterion = nn.CrossEntropyLoss()
    instrument_criterion = nn.BCEWithLogitsLoss(pos_weight=instrument_weight)
    action_criterion = nn.BCEWithLogitsLoss(pos_weight=action_weight)

    step = 0
    # ipdb.set_trace()

    # Train loop
    while step < max_step_num:
        for _, data in enumerate(train_loader):
            if step % log_freq == 0 and step != 0:
            # if step % log_freq == 0:
                print("begin test...")
                model.eval()
                test(model, test_loader, sample_step, step, all_dict,
                     result_dict, test_writer, test_log_dir, naming, 0, model2)
                torch.save(model.state_dict(),
                           os.path.join(model_dir, 'model'))
                test(model, eval_loader, sample_step, step, all_dict,
                     result_dict, eval_writer, eval_log_dir, naming, 1, model2)

            model.train()
            gt_phase = data['phase'].cuda().long()
            gt_instrument = data['instrument'].cuda().float()
            gt_action = data['action'].cuda().float()
            fusion_mode = data['fusion_mode'][0]

            # ipdb.set_trace()

            if fusion_mode == 0 or fusion_mode == 1 or fusion_mode == 3 or \
                            fusion_mode == 4 or fusion_mode == 6:
                feature = data['feature'].cuda().float()
                pred_phase, pred_instrument, pred_action = model(feature)

            else:
                rgb_feature = data['rgb_feature'].cuda().float()
                flow_feature = data['flow_feature'].cuda().float()
                pred_phase1, pred_instrument1, pred_action1 = model(rgb_feature)
                pred_phase2, pred_instrument2, pred_action2 = model2(
                    flow_feature)
                pred_phase = (pred_phase1 + pred_phase2) / 2
                pred_instrument = (pred_instrument1 + pred_instrument2) / 2
                pred_action = (pred_action1 + pred_action2) / 2

            # output: batch_size x time_step x class
            loss_train_phase = 0.0
            loss_train_instrument = 0.0
            loss_train_action = 0.0
            corloss_train_action_instrument = 0.0
            corloss_train_phase = 0.0
            for i in range(batch_size):
                loss_train_phase += phase_criterion(pred_phase[i],
                                                    gt_phase[i].squeeze())
                loss_train_instrument += instrument_criterion(
                    pred_instrument[i], gt_instrument[i])
                loss_train_action += action_criterion(pred_action[i],
                                                      gt_action[i])
                corloss_train_action_instrument += \
                    get_instrument_action_correlation(
                    pred_instrument[i], pred_action[i],
                    all_dict['instrument_action_mean'], all_dict['instrument_action_std'],
                    all_dict['action_instrument_mean'], all_dict['action_instrument_std'],
                    mode=cor_mode)
                corloss_train_phase += \
                    get_phase_correlation(pred_phase[i],
                                          all_dict['phase_phase_non0diag_mean'],
                                          all_dict['phase_phase_non0diag_std'],
                                          mode=cor_mode)

            loss_train_phase = loss_train_phase / batch_size
            loss_train_instrument = loss_train_instrument / batch_size
            loss_train_action = loss_train_action / batch_size
            loss_train_total = loss_train_phase + loss_train_instrument + loss_train_action
            corloss_train_action_instrument /= batch_size
            corloss_train_phase /= batch_size

            # print and save calculation
            #####################################
            print('{} -- Step {}: Loss_total-{}'.format(naming, step,
                                                        loss_train_total))
            print('{} -- Step {}: CorLoss_Ac_Ins-{}'.format(naming, step,
                                                            corloss_train_action_instrument))
            print('{} -- Step {}: CorLoss_Phase-{}'.format(naming, step,
                                                           corloss_train_phase))
            str_list = ['loss_train_phase', 'loss_train_instrument',
                        'loss_train_action']
            info_list = [loss_train_phase, loss_train_instrument,
                         loss_train_action]
            board_info(str_list, info_list, step, result_dict, train_writer, 0)
            result_dict['corloss_train_action_instrument'].append(
                corloss_train_action_instrument.item())
            train_writer.add_scalar('loss/train_cor_action_instrument',
                                    corloss_train_action_instrument.item(),
                                    step)
            result_dict['corloss_train_phase'].append(
                corloss_train_phase.item())
            train_writer.add_scalar('loss/train_cor_phase',
                                    corloss_train_phase.item(), step)
            #####################################

            optimizer.zero_grad()
            (loss_train_total + corloss_train_action_instrument +
             corloss_train_phase).backward()
            optimizer.step()

            step += 1
            if step >= max_step_num:
                break
    np.save(os.path.join(log_dir, naming + '.npy'), result_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--feature_type', type=str)
    parser.add_argument('--fusion_mode', type=int)
    parser.add_argument('--use_tf_log', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--concurrence_weight', type=float)
    parser.add_argument('--weighted', type=int)

    args = parser.parse_args()

    feature_type = args.feature_type
    fusion_mode = args.fusion_mode
    use_tf_log = args.use_tf_log
    learning_rate = args.learning_rate
    concurrence_weight = args.concurrence_weight
    weighted = args.weighted

    if weighted == 0:
        instrument_weight = None
        action_weight = None
    else:
        instrument_weight = torch.Tensor(instrument_weight).cuda().float()
        action_weight = torch.Tensor(action_weight).cuda().float()

    case_splits = np.load('metas/case_splits.npy', allow_pickle=True).item()
    for repeat_id in range(split_repeat):
        for split_id in range(split_num):
            train_cases = list(case_splits[repeat_id])
            train_cases.pop(split_id)
            train_list = np.concatenate(train_cases)
            test_list = case_splits[repeat_id][split_id]
            eval_list = np.array([train_list[0], train_list[-1]])

            # train_list = np.array([1])
            # val_list = np.array([1])
            # test_list = np.array([2])

            eval_video = []
            test_video = []
            eval_video.append('Hei-Chole' + str(train_list[0]))
            eval_video.append('Hei-Chole' + str(train_list[-1]))
            test_video.append('Hei-Chole' + str(test_list[0]))
            test_video.append('Hei-Chole' + str(test_list[-1]))

            naming = '{}-{}-fusmode_{}-lr_{}-weighted_{}-conweight_{}' \
                     '-repeat_{}-split_{}'.format(
                naming_prefix, feature_type, fusion_mode, learning_rate,
                weighted, concurrence_weight, repeat_id, split_id)

            print('Feature_type:    {}'.format(feature_type))
            print('TFLog:           {}'.format(use_tf_log))
            print('Naming:          {}'.format(naming))

            feature_type_list = feature_type.split('_')
            if fusion_mode == 0 or fusion_mode == 2:
                input_dim = 1024
            elif fusion_mode == 1 or fusion_mode == 3 or fusion_mode == 5 or \
                            fusion_mode == 7:
                input_dim = 2048
            else:
                input_dim = 3072

            if feature_type == 'rgb' or feature_type == 'flow':
                sample_step = 16
            else:
                sample_step = 4

            model2 = None
            if model_type == 'TCN':
                model = TCNNet(input_dim=input_dim,
                               dropout_rate=dropout_rate).cuda()

            elif model_type == 'GRU':
                model = GRUNet(input_dim=input_dim,
                               dropout_rate=dropout_rate,
                               sample_step=sample_step).cuda()
                if fusion_mode == 2 or fusion_mode == 5 or fusion_mode == 7:
                    model2 = GRUNet(input_dim=1024,
                                    dropout_rate=dropout_rate,
                                    sample_step=sample_step).cuda()

            elif model_type == 'MLP':
                model = MLPNet(input_dim=input_dim,
                               dropout_rate=dropout_rate,
                               sample_step=sample_step,
                               concurrence_weight=concurrence_weight).cuda()
                if fusion_mode == 2 or fusion_mode == 5 or fusion_mode == 7:
                    model2 = MLPNet(input_dim=1024,
                                    dropout_rate=dropout_rate,
                                    sample_step=sample_step,
                                    concurrence_weight=concurrence_weight).cuda()

            else:
                raise Exception('Unknown Model Type.')

            train_name_list = num2name(train_list, feature_type_list[0])
            test_name_list = num2name(test_list, feature_type_list[0])
            eval_name_list = num2name(eval_list, feature_type_list[0])
            feature_dir = '../i3d_new/' + feature_type

            train_datadict1 = get_datadict(
                gt_root_dir='../New_Annotations',
                feature_dir=feature_dir,
                feature_files=train_name_list,
                sample_step=sample_step
            )
            test_datadict1 = get_datadict(
                gt_root_dir='../New_Annotations',
                feature_dir=feature_dir,
                feature_files=test_name_list,
                sample_step=sample_step
            )
            eval_datadict1 = get_datadict(
                gt_root_dir='../New_Annotations',
                feature_dir=feature_dir,
                feature_files=eval_name_list,
                sample_step=sample_step
            )
            train_datadict2 = None
            test_datadict2 = None
            eval_datadict2 = None
            if fusion_mode == 1 or fusion_mode == 2:
                feature_type_list[0] = 'flow'
                train_name_list = num2name(train_list, feature_type_list[0])
                test_name_list = num2name(test_list, feature_type_list[0])
                eval_name_list = num2name(eval_list, feature_type_list[0])
                feature_dir = '../i3d_new/' + '_'.join(feature_type_list)
                train_datadict2 = get_datadict(
                    gt_root_dir='../New_Annotations',
                    feature_dir=feature_dir,
                    feature_files=train_name_list,
                    sample_step=sample_step
                )
                test_datadict2 = get_datadict(
                    gt_root_dir='../New_Annotations',
                    feature_dir=feature_dir,
                    feature_files=test_name_list,
                    sample_step=sample_step
                )
                eval_datadict2 = get_datadict(
                    gt_root_dir='../New_Annotations',
                    feature_dir=feature_dir,
                    feature_files=eval_name_list,
                    sample_step=sample_step
                )

            all_dict = get_statistics(train_datadict1)

            train_dataset = TrainDataset(fusion_mode, clip_len, sample_step,
                                         train_datadict1, train_datadict2)
            test_dataset = TestDataset(fusion_mode, clip_len, test_datadict1,
                                       test_datadict2)
            eval_dataset = TestDataset(fusion_mode, clip_len, eval_datadict1,
                                       eval_datadict2)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       drop_last=True)
            eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=1,
                                                      shuffle=True)  # FALSE
            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=1,
                                                      shuffle=True)

            train(model, train_loader, eval_loader, test_loader, naming,
                  use_tf_log, sample_step, all_dict, model2)



            # MACRO
            # ENHENCED MLP

            # "action_weight": [81.29, 0.27, 336.07, 265.79],
            # "instrument_weight": [0.53, 26.45, 1.04, 38.75, 12.83, 9.18],


            # ['Hei-Chole19-rgb.npz', 'Hei-Chole24-rgb.npz', 'Hei-Chole3-rgb.npz', 'Hei-Chole20-rgb.npz', 'Hei-Chole6-rgb.npz', 'Hei-Chole12-rgb.npz', 'Hei-Chole7-rgb.npz', 'Hei-Chole17-rgb.npz', 'Hei-Chole5-rgb.npz', 'Hei-Chole21-rgb.npz', 'Hei-Chole4-rgb.npz', 'Hei-Chole9-rgb.npz', 'Hei-Chole1-rgb.npz', 'Hei-Chole10-rgb.npz', 'Hei-Chole22-rgb.npz', 'Hei-Chole13-rgb.npz']

            # ['Hei-Chole16-rgb.npz', 'Hei-Chole18-rgb.npz', 'Hei-Chole14-rgb.npz', 'Hei-Chole23-rgb.npz', 'Hei-Chole15-rgb.npz', 'Hei-Chole8-rgb.npz', 'Hei-Chole11-rgb.npz', 'Hei-Chole2-rgb.npz']

            # ['Hei-Chole19-rgb.npz', 'Hei-Chole13-rgb.npz']

            # "action_weight": [81.29, 0.27, 336.07, 265.79],
            # "instrument_weight": [0.53, 26.45, 1.04, 38.75, 12.83, 9.18],
            # # [1.5, 97.6, 0.3, 0.5]
            # # [47.4, 2.6, 35.6, 1.8, 5.2, 7.1]