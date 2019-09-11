import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import ipdb
from config import *


def get_datadict(
        gt_root_dir,
        feature_dir,
        feature_files,
        sample_step
):
    print('Loading Data...')
    print(feature_files)

    features = [np.load(os.path.join(feature_dir, i))
                for i in feature_files]

    video_names = [i['video_name'].item().decode('UTF-8') for i in features]
    frame_nums = [i['frame_cnt'].item() for i in features]
    features = [i['feature'] for i in features]

    video_num = len(video_names)

    all_gts = [{} for i in range(video_num)]

    for i in range(video_num):

        for gt_type in ['action', 'action_detailed',
                        'phase', 'instrument', 'instrument_detailed']:
            gt_dir = os.path.join(gt_root_dir,
                                  gt_type.split('_')[0].capitalize())

            gt_file = os.path.join(gt_dir, video_names[i].lower())
            gt_file = '{}_annotation_{}.csv'.format(gt_file, gt_type)

            gt_data = np.loadtxt(gt_file, delimiter=',')
            gt_data = gt_data[:, 1:]
            if gt_type == 'instrument':
                gt_data = gt_data[:, 0:7]

            all_gts[i][gt_type] = gt_data

    ################# CHECK #########

    print('Check Data...')

    gt_lens = []

    for i in range(video_num):

        gt_len = np.unique([v.shape[0] for v in all_gts[i].values()])
        assert (gt_len.size == 1)
        gt_len = gt_len.item()
        gt_lens.append(gt_len)

        frame_num = frame_nums[i]
        feature_len = features[i].shape[1]

        assert ((frame_num - feature_len * sample_step) >= 0)
        assert((frame_num -  feature_len * sample_step) < 16)

        if frame_num != (gt_len - 1):
            print('[WARNING] Inconsistent Frame! {}'.format(video_names[i]))
            print('GT: {}, Frame: {}, Feature: {}'.format(gt_len, frame_num,
                                                          feature_len))
    #################

    datadict = {
        'all_gts': all_gts,
        'gt_lens': gt_lens,
        'video_names': video_names,
        'frame_nums': frame_nums,
        'features': features
    }

    return datadict


class TrainDataset(Dataset):
    def __init__(self, datadict, clip_len, sample_step):

        self.clip_len = clip_len
        self.sample_step = sample_step

        self.all_gts = datadict['all_gts']
        self.gt_lens = datadict['gt_lens']
        self.video_names = datadict['video_names']
        self.frame_nums = datadict['frame_nums']
        self.features = datadict['features']

        self.video_num = len(self.video_names)

        for f in self.features:
            if f.shape[1] < clip_len:
                raise Exception('Clip Length Too Large.')

    def __len__(self):
        return 100000

    def __getitem__(self, idx):

        video_idx = random.randint(0, self.video_num - 1)  # <=  <=

        all_gt = self.all_gts[video_idx]
        gt_len = self.gt_lens[video_idx]
        video_name = self.video_names[video_idx]
        frame_num = self.frame_nums[video_idx]
        feature = self.features[video_idx]

        feature_start = random.randint(0, feature.shape[1] - self.clip_len)
        feature_end = feature_start + self.clip_len

        gt_start = int(
            np.floor(gt_len * feature_start * self.sample_step / frame_num))
        gt_end = int(gt_start + self.clip_len * self.sample_step)

        # print(feature_start, feature_end, gt_start, gt_end)
        return_dict = {}
        return_dict['gt_len'] = gt_len
        return_dict['video_name'] = video_name
        return_dict['frame_num'] = frame_num
        return_dict['feature'] = random.choice(
            [i for i in feature[:, feature_start:feature_end, :]])

        assert (return_dict['feature'].shape[0] == self.clip_len)

        for key in all_gt.keys():
            return_dict[key] = all_gt[key][gt_start:gt_end, :]

            assert (return_dict[key].shape[0] == self.clip_len * self.sample_step)

        return return_dict


class TestDataset(Dataset):
    def __init__(self, datadict, clip_len):

        self.clip_len = clip_len

        self.all_gts = datadict['all_gts']
        self.gt_lens = datadict['gt_lens']
        self.video_names = datadict['video_names']
        self.frame_nums = datadict['frame_nums']
        self.features = datadict['features']

        self.video_num = len(self.video_names)

        for f in self.features:
            if f.shape[1] < clip_len:
                raise Exception('Clip Length Too Large.')

    def __len__(self):
        return self.video_num

    def __getitem__(self, idx):

        return_dict = {}
        return_dict['gt_len'] = self.gt_lens[idx]
        return_dict['video_name'] = self.video_names[idx]
        return_dict['frame_num'] = self.frame_nums[idx]
        return_dict['feature'] = self.features[idx]

        for key in self.all_gts[idx].keys():
            return_dict[key] = self.all_gts[idx][key]

        return return_dict
