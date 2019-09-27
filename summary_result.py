import os
import numpy as np
import ipdb

from config import *

locals().update(training_params)

log_root_dir = './logs'
feature_files = ['rgb-0', 'flow-0', 'rgb-1', 'rgb-2', 'rgb_oversample_4-0',
                 'flow_oversample_4-0', 'rgb_oversample_4-1',
                 'rgb_oversample_4-2', 'rgb_oversample_4_norm-0',
                 'flow_oversample_4_norm-0', 'rgb_oversample_4_norm-1',
                 'rgb_oversample_4_norm-2']
models = ['GRU', 'MLP', 'TCN']
settings = ['cat2phase']
results = ['loss_train_phase', 'loss_train_instrument', 'loss_train_action',
           'loss_test_phase', 'loss_test_instrument', 'loss_test_action',
           'acc_test_phase', 'acc_test_instrument', 'acc_test_action',
           'prec_test_instrument', 'prec_test_action', 'rec_test_instrument',
           'rec_test_action', 'f1_test_instrument', 'f1_test_action',
           'mAP_test_instrument', 'mAP_test_action']
multi_vals = ['05']

feature_num = len(feature_files)
model_num = len(models)
setting_num = len(settings)
result_num = len(results)
multi_val_num = len(multi_vals)

metrics_data = np.zeros(
    (split_repeat, split_num, feature_num, model_num, setting_num, multi_val_num, result_num))

for repeat_id in range(split_repeat):
    for split_id in range(split_num):
        for feature_id, feature_file in enumerate(feature_files):
            for model_id, model in enumerate(models):
                for setting_id, setting in enumerate(settings):
                    for val_id, multi_val in enumerate(multi_vals):

                        naming = '{}-{}-val{}-{}-repeat_{}-split_{}'.format(
                            model, setting, multi_val, feature_file, repeat_id, split_id)

                        if naming not in os.listdir(log_root_dir):
                            #print(naming)
                            #print("No this directory!")
                            continue
                        if naming + '.npy' not in os.listdir(os.path.join(log_root_dir, naming)):
                            #print(naming)
                            #print("No npy file in this directory!")
                            continue
                        data_file = os.path.join(log_root_dir, naming, naming+'.npy')
                        data = np.load(data_file, allow_pickle=True).item()

                        for result_id, result in enumerate(results):
                            if result.split('_')[0] == 'loss':
                                result_data = (-1) * np.array(data[result])[-1]
                                result_data = result_data.cpu().detach().numpy()
                            else:
                                result_data = np.array(data[result])[-1]
                            if np.isnan(result_data):
                                continue
                            print(result, result_data)
                            metrics_data[repeat_id, split_id, feature_id, model_id, setting_id, val_id, result_id] \
                                = result_data


metrics_mean = metrics_data.sum(-1).mean(0).mean(0)
metrics_data = metrics_data.reshape(split_repeat*split_num, feature_num,
                                    model_num, setting_num, multi_val_num,
                                    result_num)
metrics_variation = metrics_data.sum(-1).std(0)

print(metrics_mean[0, 0, :, :])
print(metrics_variation[0, 0, :, :])
