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

def train(model, loss, optimizer, x_val, y_val):
    x = Variable(x_val, requires_grad=False)
    y = Variable(y_val, requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x)
    output = loss.forward(fx, y)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    return output.item()

def predict(model, x_val):
    x = Variable(x_val, requires_grad=False)
    output = model.forward(x)
    return output.data.numpy().argmax(axis=1)
