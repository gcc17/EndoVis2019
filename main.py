import torch
import numpy as np
import torch.nn as nn
import argparse
import torch.optim as optim
from my_dataset import TrainDataset, TestDataset
from model import TCNNet, GRUNet
import os
from config import *
from utils import get_phase_error, get_instrument_error, get_action_error
import pdb
######################

locals().update(training_params)

######################

# test的损失函数怎么表示？每一帧的loss的平均？
def test(model, test_loader):

    model.eval()
    test_phase = []
    gt_phase = []
    test_instrument = []
    gt_instrument = []
    test_action = []
    gt_action = []

    with torch.no_grad:
        for _, data in enumerate(test_loader):

            feature = data['data'].cuda().float()
            pred_phase, pred_instrument, pred_action = model(feature)
            test_phase.append(pred_phase)
            gt_phase.append(data['phase'].cuda().float())
            test_instrument.append(pred_instrument)
            gt_instrument.append(data['instrument'].cuda().float())
            test_action.append(pred_action)
            gt_action.append(data['action'].cuda().float())

    return get_phase_error(test_phase, gt_phase), get_instrument_error(test_instrument, gt_instrument), get_action_error(test_action, gt_action)


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
    # 因为用的是交叉熵损失函数，阶段的预测也要用one-hot
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    optimizer.zero_grad()

    phase_criterion = nn.CrossEntropyLoss()
    instrument_criterion = nn.MultiLabelSoftMarginLoss()
    action_criterion = nn.MultiLabelSoftMarginLoss()

    step = 0

    # Train loop
    for _, data in enumerate(train_loader):

        if step % log_freq == 0:

            model.eval()
            loss_test_phase, loss_test_instrument, loss_test_action = test(
                model, test_loader)
            print('{} test_phase:{} test_instrument:{} test_action:{}'.format(
                naming, loss_test_phase, loss_test_instrument, loss_test_action))

            if use_tf_log:
                logger.scalar_summary('test_phase', loss_test_phase, step)
                logger.scalar_summary('test_instrument', loss_test_instrument,
                                      step)
                logger.scalar_summary('test_action', loss_test_action, step)
            result_dict['test_phase'].append(loss_test_phase)
            result_dict['test_instrument'].append(loss_test_instrument)
            result_dict['test_action'].append(loss_test_action)

            torch.save(model.state_dict(), os.path.join(model_dir, 'model'),)

        model.train()
        gt_phase = data['phase'].cuda().float()
        gt_instrument = [data['instrument'].cuda().float()]
        gt_action = [data['action'].cuda().float()]
        feature = data['full_data'].cuda().float()

        pdb.set_trace()

        pred_phase, pred_instrument, pred_action = model(feature)
        loss_phase = phase_criterion(pred_phase, gt_phase)
        loss_phase = loss_phase.mean()
        loss_instrument = instrument_criterion(pred_instrument, gt_instrument)
        loss_instrument = loss_instrument.mean()
        loss_action = action_criterion(pred_action, gt_action)
        loss_action = loss_action.mean()
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
    np.save(os.path.join(log_dir, naming+'.npy'), result_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--feature_file', type=str)
    parser.add_argument('--use_tf_log', type=int)

    args = parser.parse_args()

    feature_file = args.feature_file
    use_tf_log = args.use_tf_log

    # load all data!
    all_data = np.load(feature_file, allow_pickle=True).item()

    for repeat_id in range(split_repeat):
        for split_id in range(split_num):
            naming = '{}-{}-repeat_{}-split_{}'.format(naming_prefix, feature_file.split('/')[-1], repeat_id, split_id)
            print('Repeat:      {}'.format(repeat_id))
            print('Split:       {}'.format(split_id))
            print('Feature:     {}'.format(feature_file))
            print('TFLog:       {}'.format(use_tf_log))
            print('Naming:      {}'.format(naming))

            if model_type == 'TCN':
                model = TCNNet(input_dim=input_dim, dropout_rate=dropout_rate).cuda()

            elif model_type == 'GRU':
                model = GRUNet(input_dim=input_dim, dropout_rate=dropout_rate).cuda()

            else:
                raise Exception('Unknown Model Type.')

            train_dataset = TrainDataset(repeat_id, split_id, all_data)
            test_dataset = TestDataset(repeat_id, split_id, all_data)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            train(model, train_loader, test_loader, naming, use_tf_log)