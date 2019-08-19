#************************************************************************
#    > File Name:     dataset.py
#    > Author:        Guo_mq
#    > Mail:          836918658@qq.com 
#    > Created Time:  Thu 15 Aug 2019 10:55:20 AM CST
#************************************************************************

import numpy as np
from torch.utils.data import Dataset

from utils import get_test_cases, get_train_cases

class TestDataset(Dataset):
    def __init__(self, feature_name, feature_type='rgb', length=512):

        self.feature_name = feature_name
        self.feature_type = feature_type
        self.test_cases = get_test_cases(feature_name, feature_type, length)

    def __len__(self):
        return len(self.test_cases)

    def __getitem__(self, idx):

        return_dict = {}
        return_dict['idx'] = np.array(idx)
        return_dict['gt_score'] = get_gt(idx)
        return_dict['data'] = np.array(self.test_cases[idx])
        return_dict['is_test_case'] = 1

        return return_dict


class TrainDataset(Dataset):
    def __init__(self, feature_name, feature_type='rgb', length=512):

        self.feature_name = feature_name
        self.feature_type = feature_type
        self.train_cases = get_train_cases(feature_name, feature_type, length)

    def __len__(self):
        return len(self.test_cases)

    def __getitem__(self, idx):
        return_dict = {}
        return_dict['idx'] = np.array(idx)
        return_dict['phase'], return_dict['instrument'], return_dict[
            'action'], return_dict['action_detailed'] = get_gt(
            self.feature_name[idx])
        return_dict['data'] = np.array(self.train_cases[idx])
        return_dict['is_train_case'] = 1

        return return_dict

