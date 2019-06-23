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
    self.embed = nn.Embedding(num_embeddings=21, embedding_dim=self.args.input_size, padding_idx=20)
    self.densel1 = nn.Linear(self.args.input_size, self.args.n_l1)
    self.densel2 = nn.Linear(self.args.n_l1, self.args.n_l1)
    self.bi_rnn = nn.LSTM(input_size=self.args.n_l1+self.args.input_size, hidden_size=self.args.n_rnn_hid, num_layers=3, bidirectional=True, batch_first=True)
    self.drop = nn.Dropout(p=0.5)
    self.relu = nn.ReLU()

    self.densel3 = nn.Linear(self.args.n_rnn_hid*2, self.args.n_l2)
    self.densel4 = nn.Linear(self.args.n_l2, self.args.n_l2)

    self.label = nn.Linear(self.args.n_l2, self.args.n_outputs)

    if self.args.crf:
        self.crf = CRF(num_tags=self.args.n_outputs, batch_first=True).double()

    self.init_weights()

    
  def init_weights(self):
    self.densel1.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.densel1.weight.data, gain=1.0)

    self.densel2.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.densel2.weight.data, gain=1.0)

    self.densel3.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.densel3.weight.data, gain=1.0)

    self.densel4.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.densel4.weight.data, gain=1.0)

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
    inp = inp.long()
    inp = self.embed(inp)
    x = self.relu(self.densel2(self.relu(self.densel1(inp))))
    x = torch.cat((inp,x), dim=2)
    pack = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
    packed_output, _ = self.bi_rnn(pack)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

    output = self.relu(self.densel4(self.drop(self.relu(self.densel3(self.drop(output))))))
    out = self.label(output)

    return out