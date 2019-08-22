import random
import numpy as np
from torch.utils.data import Dataset
from config import *

from utils import get_test_cases, get_train_case, get_test_gt, get_train_gt


class TestDataset(Dataset):
    def __init__(self, feature_name, feature_type='rgb', length=512):
        self.feature_name = feature_name
        self.feature_type = feature_type
        self.length = length
        self.test_cases, self.flips_nums = get_test_cases(feature_name,
                                                          feature_type, length)

    # for test
    # for i in range(len(feature_name)):
    #    get_test_gt(feature_name[i], 2, length)

    def __len__(self):
        return len(self.test_cases)

    def __getitem__(self, idx):
        return_dict = {}
        return_dict['idx'] = np.array(idx)
        return_dict['gt'] = get_test_gt(self.feature_name[idx],
                                        self.flips_nums[idx], self.length)
        return_dict['data'] = np.array(self.test_cases[idx])
        return_dict['is_test_case'] = 1
        return_dict['video_clip'] = self.flips_nums[idx]

        return return_dict


class TrainDataset(Dataset):
    def __init__(self, feature_names, feature_type='rgb', length=512):
        self.feature_names = feature_names
        self.feature_type = feature_type
        self.length = length

    # for test
    # for i in range(len(feature_names)):
    #    get_train_gt(feature_names[i], 0, length)

    def __len__(self):
        return len(self.feature_names)*10 # TODO

    def __getitem__(self, idx):
        data, name, frame = get_train_case(self.feature_names,
                                           self.feature_type, self.length)

        return_dict = {}
        return_dict['idx'] = np.array(idx)
        return_dict['gt_phase'], return_dict['gt_instrument'], return_dict[
            'gt_action'], return_dict['gt_action_detailed'] = get_train_gt(
            name, frame * i3d_time, self.length)
        return_dict['data'] = np.array(data)
        return_dict['is_train_case'] = 1
        return_dict['start_frame'] = frame * i3d_time

        return return_dict
