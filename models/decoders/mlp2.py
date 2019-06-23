import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Decoder(nn.Module):

  def __init__(self, args):
    super().__init__()
    self.args = args
    self.densel1 = nn.Linear(self.args.n_rnn_hid*2, self.args.n_l2)
    self.densel2 = nn.Linear(self.args.n_l2, self.args.n_l2)
    self.drop = nn.Dropout(p=0.5)
    self.relu = nn.ReLU()

    self.label = nn.Linear(self.args.n_l2, self.args.n_outputs)

    self.init_weights()

    
  def init_weights(self):
    self.densel1.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.densel1.weight.data, gain=1.0)

    self.densel2.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.densel2.weight.data, gain=1.0)

    self.label.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.label.weight.data, gain=1.0)
    
  def forward(self, inp, seq_lengths):
    output = self.relu(self.densel2(self.drop(self.relu(self.densel1(self.drop(inp))))))
    out = self.label(output)

    return out