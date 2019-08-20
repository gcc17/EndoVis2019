import torch
import numpy as np
import torch.nn as nn
import argparse
import torch.optim as optim
from my_dataset import TrainDataset, TestDataset
from model import TCNNet, GRUNet
import os
from config import *
from utils import get_phase_error, get_instrument_error, get_action_error, num2name
import ipdb

######################

locals().update(training_params)

######################


def test(model, test_loader):
    model.eval()
    test_num = 0
    total_phase = 0.0
    total_instrument = 0.0
    total_action = 0.0

    with torch.no_grad():
        for num, data in enumerate(test_loader):

            feature = data['data'].cuda().float()
            is_test_case = data['is_test_case'].item()
            video_clip = data['video_clip']

            if is_test_case:
                phase_error = 0.0
                instrument_error = 0.0
                action_error = 0.0
                for i in range(video_clip):
                    feature_clip = feature.squeeze(0)[i].unsqueeze(0)
                    # print(feature_clip.shape)
                    pred_phase, pred_instrument, pred_action = model(feature_clip)
                    phase_error += get_phase_error(pred_phase.squeeze(0),
                                    data['gt'][i]['gt_phase'].squeeze(0).cuda().long())
                    instrument_error += get_instrument_error(pred_instrument.squeeze(0),
                                    data['gt'][i]['gt_instrument'].squeeze(0).cuda().float())
                    action_error += get_action_error(pred_action.squeeze(0),
                                    data['gt'][i]['gt_action'].squeeze(0).cuda().float())
                    # print("phase_error is ", phase_error, "instrument_error
                    # is ",instrument_error,"action_error is ", action_error)
                # print("\n")
                test_num += 1
                phase_error = phase_error / video_clip.item()
                instrument_error = instrument_error / video_clip.item()
                action_error = action_error / video_clip.item()
                total_phase += phase_error
                total_instrument += instrument_error
                total_action += action_error

    return total_phase / test_num, total_instrument / test_num, total_action / test_num


def train(model, train_loader, test_loader, naming, use_tf_log):
    model_dir = os.path.join('./models', naming)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_dir = os.path.join('./logs', naming)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if use_tf_log:
        from logger import Logger
        logger = Logger(log_dir)

    result_dict = {'loss_phase': [], 'loss_instrument': [], 'loss_action': [],
                   'loss_total': [],
                   'test_phase': [], 'test_instrument': [], 'test_action': []}

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
            print(data['data'].shape)

            if step % log_freq == 0:

                print("begin test...")
                model.eval()
                loss_test_phase, loss_test_instrument, loss_test_action = test(
                    model, test_loader)
                print(
                    '{} test_phase:{} test_instrument:{} test_action:{}'.format(
                        naming, loss_test_phase, loss_test_instrument,
                        loss_test_action))

                if use_tf_log:
                    logger.scalar_summary('test_phase', loss_test_phase, step)
                    logger.scalar_summary('test_instrument',
                                          loss_test_instrument,
                                          step)
                    logger.scalar_summary('test_action', loss_test_action, step)
                result_dict['test_phase'].append(loss_test_phase)
                result_dict['test_instrument'].append(loss_test_instrument)
                result_dict['test_action'].append(loss_test_action)

                torch.save(model.state_dict(),
                           os.path.join(model_dir, 'model'), )

            model.train()
            gt_phase = data['gt_phase'].cuda().long()
            gt_instrument = data['gt_instrument'].cuda().float()
            gt_action = data['gt_action'].cuda().float()
            feature = data['data'].cuda().float()

            # ipdb.set_trace()

            pred_phase, pred_instrument, pred_action = model(feature)
            # output: batch_size x time_step x class
            loss_phase = 0.0
            loss_instrument = 0.0
            loss_action = 0.0
            for i in range(batch_size):
                loss_phase += phase_criterion(pred_phase[i], gt_phase[i].squeeze())
                loss_instrument += instrument_criterion(pred_instrument[i],
                                                   gt_instrument[i])
                loss_action += action_criterion(pred_action[i], gt_action[i])
            loss_phase = loss_phase/ batch_size
            loss_instrument = loss_instrument / batch_size
            loss_action = loss_action / batch_size
            loss_total = loss_phase + loss_instrument + loss_action

            # print and save calculation
            #####################################
            print('{} -- Step {}: Loss_total-{}'.format(naming, step,
                                                        loss_total.item()))

            result_dict['loss_total'].append(loss_total.item())
            result_dict['loss_phase'].append(loss_phase.item())
            result_dict['loss_instrument'].append(loss_instrument.item())
            result_dict['loss_action'].append(loss_action.item())
            if use_tf_log:
                logger.scalar_summary('loss_total', loss_total.item(), step)
                logger.scalar_summary('loss_phase', loss_phase.item(), step)
                logger.scalar_summary('loss_instrument', loss_instrument.item(),
                                      step)
                logger.scalar_summary('loss_action', loss_action.item(), step)
            #####################################

            optimizer.zero_grad()
            loss_total.backward()
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
    parser.add_argument('--train', action='append')
    parser.add_argument('--test', action='append')


    args = parser.parse_args()

    feature_type = args.feature_type
    use_tf_log = args.use_tf_log
    train_list = args.train
    test_list = args.test
    train_name_list = num2name(train_list, feature_type)
    test_name_list = num2name(test_list, feature_type)

    naming = naming_prefix + "-train"
    for i in range(len(train_list)):
        naming = naming + '_' + train_list[i]
    naming = naming + '-test'
    for i in range(len(test_list)):
        naming = naming + '_' + test_list[i]
    print('Feature_type:    {}'.format(feature_type))
    print('TFLog:           {}'.format(use_tf_log))
    print('Naming:          {}'.format(naming))

    if model_type == 'TCN':
        model = TCNNet(input_dim=input_dim, dropout_rate=dropout_rate).cuda()

    elif model_type == 'GRU':
        model = GRUNet(input_dim=input_dim, dropout_rate=dropout_rate).cuda()

    else:
        raise Exception('Unknown Model Type.')

    train_dataset = TrainDataset(train_name_list, feature_type)
    test_dataset = TestDataset(test_name_list, feature_type)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    train(model, train_loader, test_loader, naming, use_tf_log)
