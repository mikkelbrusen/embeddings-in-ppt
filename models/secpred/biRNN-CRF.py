import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.secpred.base import Model as BaseModel
from models.secpred.base import Config as BaseConfig
from model_utils.crf_layer import CRF

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args)
    self.Model = Model

class Model(BaseModel):
  def __init__(self, args):
    super().__init__(args)
    #First layer
    self.densel1 = nn.Linear(self.args.input_size, self.args.n_l1)
    #Second layer
    self.bi_rnn = nn.GRU(input_size=self.args.n_l1+self.args.input_size, hidden_size=self.args.n_rnn_hid, bidirectional=True, batch_first=True)
    #Third layer
    self.drop = nn.Dropout(p=0.5)
    self.relu = nn.ReLU()
    #Fourth layer
    self.label = nn.Linear(self.args.n_rnn_hid*2, self.args.n_outputs)

    if self.args.crf:
        #Fifth layer
        self.crf = CRF(num_tags=self.args.n_outputs, batch_first=True).double()

    self.init_weights()

    
  def init_weights(self):
    self.densel1.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.densel1.weight.data, gain=1.0)

    self.label.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.label.weight.data, gain=1.0)

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
    #First layer
    x = self.densel1(inp)
    x = self.relu(x)
    x = torch.cat((inp,x), dim=2)
    #Second layer
    pack = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
    packed_output, _ = self.bi_rnn(pack)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
    #Third layer
    output = self.drop(output)
    #Fourth layer
    out = self.label(output)

    return out
