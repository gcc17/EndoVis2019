import random
import numpy as np
from torch.utils.data import Dataset
from config import *

from utils import get_test_cases, get_train_case, get_test_gt, get_train_gt, get_test_combination_cases, get_train_combination_case


class TestDataset(Dataset):
    def __init__(self, feature_name, feature_type='rgb', length=512):
        self.feature_name = feature_name
        self.feature_type = feature_type
        self.length = length
        if (combination == "True"):
            self.test_cases, self.flips_nums = get_test_combination_cases(feature_name, feature_type, length)
        else:
            self.test_cases, self.flips_nums = get_test_cases(feature_name, feature_type, length)

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
    #    get_train_combination_case(self.feature_names, self.feature_type, self.length)

    def __len__(self):
        return len(self.feature_names)*10 # TODO

    def __getitem__(self, idx):
        if (combination == "True"):
            data, name, frame = get_train_combination_case(self.feature_names, self.feature_type, self.length)
        else:
            data, name, frame = get_train_case(self.feature_names, self.feature_type, self.length)

        return_dict = {}
        return_dict['idx'] = np.array(idx)
        return_dict['gt_phase'], return_dict['gt_instrument'], return_dict[
            'gt_action'], return_dict['gt_action_detailed'], return_dict['gt_calot_skill'], return_dict['gt_dissection_skill'] = get_train_gt(
            name, frame * i3d_time, self.length)
        return_dict['data'] = np.array(data)
        return_dict['is_train_case'] = 1
        return_dict['start_frame'] = frame * i3d_time

        return return_dict

#def main():
#    test_name = ['Hei-Chole1-rgb.npz', 'Hei-Chole2-rgb.npz', 'Hei-Chole7-rgb.npz']
#    train_name = ['Hei-Chole11-rgb.npz', 'Hei-Chole12-rgb.npz', 'Hei-Chole10-rgb.npz']
#    t = TestDataset(test_name, 'rgb_oversample_4', 512)
#    y = TrainDataset(train_name, 'flow_oversample_4', 512)

#main()
