import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import init_weights

class Encoder(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.densel1 = nn.Linear(self.args.n_features, self.args.n_l1)
    self.densel2 = nn.Linear(self.args.n_l1, self.args.n_l1)
    self.bi_rnn = nn.LSTM(input_size=self.args.n_l1+self.args.n_features, hidden_size=self.args.n_hid, num_layers=3, bidirectional=True, batch_first=True)
    self.drop = nn.Dropout(p=0.5)
    self.relu = nn.ReLU()

    init_weights(self)
    self.init_weights()
   
  def init_weights(self):
    self.densel1.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.densel1.weight.data, gain=1.0)

    self.densel2.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.densel2.weight.data, gain=1.0)
    
  def forward(self, inp, seq_lengths):
    x = self.relu(self.densel2(self.relu(self.densel1(inp))))
    x = torch.cat((inp,x), dim=2)
    pack = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
    packed_output, _ = self.bi_rnn(pack)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
    return output