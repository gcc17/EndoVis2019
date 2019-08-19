import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import ipdb

############### Embedding ######################

class EmbedModule(nn.Module):

    def __init__(self, input_dim, output_dim, dropout_rate):

        self.input_dim =input_dim
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


########### GRU ####################

class GRUNet(nn.Module):
    def __init__(self, input_dim):
        super(GRUNet, self).__init__()

        self.hidden_dim = 32
        self.input_dim = input_dim
        self.base = nn.GRU(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.phase_pred_net = nn.Sequential(
            nn.Linear(self.hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 7)
        )
        self.instrument_pred_net = nn.Sequential(
            nn.Linear(self.hidden_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 21)
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

        batch_size = x.shape[0]
        frame_num = x.shape[1]

        h_state = torch.zeros(1, batch_size,self.hidden_dim).cuda().float()
        x, _ = self.base(x, h_state)

        pred_phase = self.phase_pred_net(x)
        pred_instrument = self.instrument_pred_net(x)
        pred_action = self.action_pred_net(x)

        return pred_phase, pred_instrument, pred_action