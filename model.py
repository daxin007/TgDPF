import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tst import Transformer
from configs import *

class NETLSTM(nn.Module):

    def __init__(self, dim_in=None, dim_hidden=None, dim_out=None,
                 num_layer=None, dropout=None, **kwargs):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.num_layer = num_layer
        self.dropout = dropout
        self.dim_out = dim_out
        self.lstm = nn.LSTM(self.dim_in,
                            self.dim_hidden,
                            self.num_layer,
                            batch_first=True,
                            dropout=self.dropout)
        self.fc2 = nn.Linear(self.dim_hidden, int(self.dim_hidden / 2))
        self.fc3 = nn.Linear(int(self.dim_hidden / 2), self.dim_out)
        self.fc4 = nn.Linear(int(self.dim_hidden / 2), int(self.dim_hidden / 2))
        self.bn = nn.BatchNorm1d(int(self.dim_hidden / 2))
        self.device = DEVICE

    def forward(self, x, hs=None, training=True):
        if hs is None:
            h = Variable(torch.zeros(self.num_layer, len(x), self.dim_hidden, device=self.device))
            v = Variable(torch.zeros(self.num_layer, len(x), self.dim_hidden, device=self.device))
            hs = (h, v)
        out, hs_0 = self.lstm(x, hs)
        if training:
            out = out[:, -TIME_AHEAD:, :]
        out = out.contiguous()
        out = out.view(-1, self.dim_hidden)
        # out = nn.functional.relu(self.bn(self.fc2(out)))
        out = nn.functional.relu(self.fc2(out))
        # out = nn.functional.tanh(self.fc2(out))
        # out = nn.functional.relu(self.fc4(out))
        out = self.fc3(out)
        # return out
        return out, hs_0

    def set_device(self, device):
        self.device = device

