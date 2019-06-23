import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.densel1 = nn.Linear(self.args.input_size, self.args.n_l1)
    self.densel2 = nn.Linear(self.args.n_l1, self.args.n_l1)
    self.bi_rnn = nn.LSTM(input_size=self.args.n_l1+self.args.input_size, hidden_size=self.args.n_rnn_hid, num_layers=3, bidirectional=True, batch_first=True)
    self.drop = nn.Dropout(p=0.5)
    self.relu = nn.ReLU()

    self.init_weights()
   
  def init_weights(self):
    self.densel1.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.densel1.weight.data, gain=1.0)

    self.densel2.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.densel2.weight.data, gain=1.0)

    for m in self.modules():
      if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
        for name, param in m.named_parameters():
      #        if 'weight_ih' in name:
      #            torch.nn.init.orthogonal_(param.data, gain=1)
      #        elif 'weight_hh' in name:
      #            torch.nn.init.orthogonal_(param.data, gain=1)
          if 'bias_ih' in name:
              param.data.zero_()
          elif 'bias_hh' in name:
              param.data.zero_()
    
  def forward(self, inp, seq_lengths):
    x = self.relu(self.densel2(self.relu(self.densel1(inp))))
    x = torch.cat((inp,x), dim=2)
    pack = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
    packed_output, _ = self.bi_rnn(pack)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
    return output