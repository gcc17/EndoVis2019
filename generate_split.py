import numpy as np
from config import *
import ipdb

def get_case_splits():

    case_ids = range(1, case_num+1)
    case_ids = np.array(case_ids)

    np.random.shuffle(case_ids)
    case_splits = np.array_split(case_ids, split_num)

    return case_splits


if __name__ == '__main__':

    case_splits = {}
    for repeat_idx in range(split_repeat):
        case_splits[repeat_idx] = get_case_splits()

    np.save('metas/case_splits.npy', case_splits)
