import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
from config import *

##################################
instrument_concurrence = \
[[0.25, 0.04, 0.50, 0.02, 0.06, 0.12, 0],
 [0.69, 0.29, 0, 0, 0.01, 0.01, 0],
 [0.69, 0, 0.28, 0, 0.01, 0.01, 0],
 [0.61, 0, 0, 0.33, 0.01, 0, 0],
 [0.34, 0.01, 0.02, 0.01, 0.08, 0.13, 0],
 [0.69, 0, 0.02, 0.01, 0.08, 0.13, 0],
 [0.01, 0, 0, 0, 0, 0, 0.03]]
action_concurrence = \
[[0.27, 0.73, 0, 0],
 [0.01, 0.98, 0, 0.01],
 [0, 0.94, 0.02, 0],
 [0, 0.97, 0, 0.02]]
instrument_category = 7
##################################


############### Embedding ######################

class EmbedModule(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(EmbedModule, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input: Batch x Temporal x Feature Dim
        # Batch x 512 x 1024
        x = x.permute(0, 2, 1)
        x = self.dropout(x.unsqueeze(3)).squeeze(3)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        x = self.relu(x)

        return x


########### Frozen-level ######################
# using for concurrence computation
class ConNet(nn.Module):
    def __init__(self, class_num, concurrence_weight):
        super(ConNet, self).__init__()
        self.class_num = class_num
        self.concurrence_weight = concurrence_weight

        self.linear = nn.Linear(class_num, class_num, bias=False)
        if class_num == 4:
            action_matrix = (torch.eye(4)+torch.Tensor(
                action_concurrence)*concurrence_weight).cuda().float()
            self.linear.weight = nn.Parameter(action_matrix.t())
        else:
            instrument_matrix = (torch.eye(instrument_category) + torch.Tensor(
                instrument_concurrence) * concurrence_weight).cuda().float()
            self.linear.weight = nn.Parameter(instrument_matrix.t())
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.linear(x)
        return x

########### Frame-wise MLP ####################

class MLPNet(nn.Module):
    def __init__(self, input_dim, dropout_rate, sample_step,
                 concurrence_weight):
        super(MLPNet, self).__init__()

        self.input_dim = input_dim
        self.sample_step = sample_step
        self.hidden_dim = 32
        self.base = EmbedModule(input_dim, 128, dropout_rate)
        self.middle = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_dim),
            nn.ReLU(),
        )
        self.phase_pred_net = nn.Sequential(
            nn.Linear(self.hidden_dim + 11, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 7)
        )
        self.instrument_pred_net = nn.Sequential(
            nn.Linear(self.hidden_dim, 24),
            nn.ReLU(),
            nn.Linear(24, instrument_category),
            ConNet(instrument_category, concurrence_weight)
        )
        self.action_pred_net = nn.Sequential(
            nn.Linear(self.hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            ConNet(4, concurrence_weight)
        )

    def forward(self, x):
        # x (batch, time_step, input_dim)
        x = self.base(x)
        x = self.middle(x)
        pred_instrument = self.instrument_pred_net(x)
        pred_action = self.action_pred_net(x)
        tmp = torch.cat([pred_instrument, pred_action], dim=2)
        pred_phase = self.phase_pred_net(torch.cat([x, tmp], dim=2))

        upsample_phase = nn.UpsamplingBilinear2d([clip_len * self.sample_step,
                                                  7])
        upsample_instrument = nn.UpsamplingBilinear2d(
            [clip_len * self.sample_step, 7])
        upsample_action = nn.UpsamplingBilinear2d([clip_len * self.sample_step,
                                                   4])

        pred_phase = upsample_phase(pred_phase.unsqueeze(0))
        pred_phase = pred_phase.squeeze(0)
        pred_instrument = upsample_instrument(pred_instrument.unsqueeze(0))
        pred_instrument = pred_instrument.squeeze(0)
        pred_action = upsample_action(pred_action.unsqueeze(0))
        pred_action = pred_action.squeeze(0)

        return pred_phase, pred_instrument, pred_action


# model = MLPNet(1024, 1e-3, 4, 0.5)
# for name, param in model.named_parameters():
#     if param.requires_grad is False:
#         print(name)

# x = torch.randn(1, 512, 1024)
# p, i, a = model(x)
# print(p.shape)
# print(i.shape)
# print(a.shape)


########### GRU ####################

class GRUNet(nn.Module):
    def __init__(self, input_dim, dropout_rate, sample_step):
        super(GRUNet, self).__init__()

        self.hidden_dim = 32
        self.input_dim = input_dim
        self.sample_step = sample_step

        self.base = EmbedModule(input_dim, 64, dropout_rate)
        self.middle = nn.GRU(
            input_size=64,
            hidden_size=self.hidden_dim,
            num_layers=3,
            batch_first=True
        )
        self.phase_pred_net = nn.Sequential(
            nn.Linear(self.hidden_dim + 11, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 7)
        )
        self.instrument_pred_net = nn.Sequential(
            nn.Linear(self.hidden_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 7)
        )
        self.action_pred_net = nn.Sequential(
            nn.Linear(self.hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

    def forward(self, x):
        # x (batch, time_step, input_dim)
        # h_state (n_layers, batch, hidden_dim)
        # r_out (batch, time_step, hidden_dim)

        x = self.base(x)
        batch_size = x.shape[0]

        h_state = torch.zeros(3, batch_size, self.hidden_dim).cuda().float()
        x, _ = self.middle(x, h_state)

        pred_instrument = self.instrument_pred_net(x)
        pred_action = self.action_pred_net(x)
        tmp = torch.cat([pred_instrument, pred_action], dim=2)
        pred_phase = self.phase_pred_net(torch.cat([x, tmp], dim=2))

        upsample_phase = nn.UpsamplingBilinear2d([clip_len * self.sample_step,
                                                  7])
        upsample_instrument = nn.UpsamplingBilinear2d(
            [clip_len * self.sample_step, 7])
        upsample_action = nn.UpsamplingBilinear2d([clip_len * self.sample_step,
                                                   4])

        pred_phase = upsample_phase(pred_phase.unsqueeze(0))
        pred_phase = pred_phase.squeeze(0)
        pred_instrument = upsample_instrument(pred_instrument.unsqueeze(0))
        pred_instrument = pred_instrument.squeeze(0)
        pred_action = upsample_action(pred_action.unsqueeze(0))
        pred_action = pred_action.squeeze(0)

        return pred_phase, pred_instrument, pred_action


# model = GRUNet(1024, 1e-3)
# x = torch.randn(1, 512, 1024)
# p, i, a = model(x)
# print(p.shape)
# print(i.shape)
# print(a.shape)


####################### TCN #################################

class TCNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TCNEncoder, self).__init__()

        self.conv = nn.Conv1d(input_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class TCNDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TCNDecoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.ConvTranspose1d(input_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class TCNNet(nn.Module):
    def __init__(self, input_dim, dropout_rate):
        super(TCNNet, self).__init__()

        self.base = EmbedModule(input_dim, 64, dropout_rate)

        self.middle = nn.Sequential(
            TCNEncoder(64, 16),
            TCNEncoder(16, 4),
            TCNDecoder(4, 16),
            TCNDecoder(16, 32)
        )

        self.phase_branch = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 7)
        )

        self.instrument_branch = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, 21)
        )

        self.action_branch = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

        def forward(self, x):
            x = self.base(x)

            padding = 4 - (x.shape[1] % 4)
            padding = padding % 4
            if padding != 0:
                x = nn.functional.pad(x, (0, 0, 0, padding), mode='constant',
                                      value=0)
            assert (x.shape[1] % 4 == 0)

            x = x.permute(0, 2, 1)
            x = self.middle(x)
            x = x.permute(0, 2, 1)

            if padding != 0:
                x = x[:, :-padding, :]

            phase = self.phase_branch(x)
            instrument = self.instrument_branch(x)
            action = self.action_branch(x)

            return phase, instrument, action
