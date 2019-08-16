#************************************************************************
#    > File Name:     dataset.py
#    > Author:        Guo_mq
#    > Mail:          836918658@qq.com 
#    > Created Time:  Thu 15 Aug 2019 10:55:20 AM CST
#************************************************************************

import numpy as np
from torch.utils.data import Dataset

from utils import get_test_cases, get_train_cases, get_gt

class TestDataset(Dataset):
    def __init__(self, feature_name, feature_type='rgb', length=512):
        
        self.feature_name = feature_name
        self.feature_type = feature_type
        self.test_cases = get_test_cases(feature_name, feature_type, length)
        for i in range(len(feature_name)):
            get_gt(feature_name[i])

    def __len__(self):
        return len(self.test_cases)

    def __getitem__(self, idx):

        return_dict = {}
        return_dict['idx'] = np.array(idx)
        return_dict['gt_score'] = get_gt(self.feature_name[idx])
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
        return_dict['gt_score'] = get_gt(self.feature_name[idx])
        return_dict['data'] = np.array(self.train_cases[idx])
        return_dict['is_train_case'] = 1

        return return_dict

def main():
    print('Test begin')
    name_test = ['Hei-Chole1-flow.npz', 'Hei-Chole2-flow.npz']
    TestDataset(name_test, 'flow')
    
    print('Train begin')
    name_train = ['Hei-Chole4-rgb.npz', 'Hei-Chole3-rgb.npz']
    TrainDataset(name_train, 'rgb')
    
    print('self\'s test done')
    return

main()


