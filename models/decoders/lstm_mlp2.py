import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Decoder(nn.Module):

  def __init__(self, args, in_size):
    super().__init__()
    self.args = args
    self.bi_rnn = nn.LSTM(input_size=in_size, hidden_size=self.args.n_hid2, bidirectional=True, batch_first=True)
    self.densel1 = nn.Linear(self.args.n_hid2*2, self.args.n_hid3)
    self.densel2 = nn.Linear(self.args.n_hid3, self.args.n_hid3)
    self.drop = nn.Dropout(p=0.5)
    self.relu = nn.ReLU()

    self.label = nn.Linear(self.args.n_hid3, self.args.num_classes)
    
  def forward(self, inp, seq_lengths):
    pack = nn.utils.rnn.pack_padded_sequence(inp, seq_lengths, batch_first=True)
    packed_output, _ = self.bi_rnn(pack)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
    output = self.relu(self.densel2(self.drop(self.relu(self.densel1(self.drop(output))))))
    out = self.label(output)

    return out