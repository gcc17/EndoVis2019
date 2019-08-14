import numpy as np
import os
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
import subprocess
from config import *

def get_phase_error(pred_phase, gt_phase):
    criterion = nn.CrossEntropyLoss()
    pred_phase = torch.autograd.Variable(torch.FloatTensor([pred_phase]))
    gt_phase = torch.autograd.Variable(torch.FloatTensor([gt_phase]))
    loss = criterion(pred_phase, gt_phase)
    return loss.mean().item()

def get_instrument_error(pred_instrument, gt_instrument):
    criterion = nn.MultiLabelSoftMarginLoss()
    pred_instrument = torch.autograd.Variable(torch.FloatTensor([
        pred_instrument]))
    gt_instrument = torch.autograd.Variable(torch.FloatTensor([gt_instrument]))
    loss = criterion(pred_instrument, gt_instrument)
    return loss.mean()

def get_action_error(pred_action, gt_action):
    criterion = nn.MultiLabelSoftMarginLoss()
    pred_action = torch.autograd.Variable(torch.FloatTensor([pred_action]))
    gt_action = torch.autograd.Variable(torch.FloatTensor([gt_action]))
    loss = criterion(pred_action, gt_action)
    return loss.mean()