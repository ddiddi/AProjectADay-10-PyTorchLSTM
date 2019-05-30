from __future__ import division
import numpy as np

import torch
from torch.autograd import Variable
from torch import optim, nn

from data_util import load_mnist

class LSTMNet(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(LSTMNet, self).__init__()
        self.hidden_dims = nn.LSTM(input_dims, hidden_dims)
        self.linear = nn.Linear(hidden_dims,output_dims, bias=False)

    def forward(self, x):
        batch_size = x.size()[1]
        h0 = Variable(torch.zeros([1, batch_size, self.hidden_dim]), requires_grad=False)
        c0 = Variable(torch.zeros([1, batch_size, self.hidden_dim]), requires_grad=False)
        fx, _ = self.lstm.forward(x, (h0, c0))
        return self.linear.forward(fx[-1])
