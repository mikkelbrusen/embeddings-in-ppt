import math
import torch
import torch.nn as nn
from torch.autograd import Variable

class SeqPred(nn.Module):
  def __init__(self, input_size, num_units_l1, num_units_lstm, num_units_l2, number_outputs):
    super(SeqPred, self).__init__()

    self.densel1 = nn.Linear(input_size, num_units_l1)
    #self.bn1 = nn.BatchNorm1d(num_features=num_units_l1)
    #self.bi_rnn = nn.GRU(input_size=num_units_l1+input_size, hidden_size=num_units_lstm, bidirectional=True, batch_first=True)
    #self.bi_rnn = nn.LSTM(input_size=num_units_l1+input_size, hidden_size=num_units_lstm, bidirectional=True, batch_first=True)
    self.bi_rnn = nn.LSTM(input_size=num_units_l1+input_size, hidden_size=num_units_lstm, num_layers=3, bidirectional=True, batch_first=True)
    self.drop = nn.Dropout(p=0.5)
    self.relu = nn.ReLU()
    #self.softmax = nn.Softmax(dim=2)
    self.densel2 = nn.Linear(num_units_lstm*2+input_size, num_units_l2)
    #self.densel2 = nn.Linear(num_units_lstm*2, num_units_l2)
    self.label = nn.Linear(num_units_l2, number_outputs)
 
    self.init_weights()
    
    
  def init_weights(self):
    self.densel1.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.densel1.weight.data, gain=1.0)
    
    self.densel2.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.densel2.weight.data, gain=1.0)

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
    x = self.densel1(inp)#.permute(0,2,1)
    #x = self.bn1(x).permute(0,2,1)
    x = self.relu(x)
    x = torch.cat((inp,x), dim=2)
    pack = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
    packed_output, _ = self.bi_rnn(pack)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
    output = torch.cat((inp,output), dim=2)
    
    output = self.drop(output)
    #output = self.densel2(output).permute(0,2,1)
    #output = self.bn1(output).permute(0,2,1)
    #output = self.relu(output)
    output = self.relu(self.densel2(output))
    output = self.drop(output)
    out = self.label(output)
    return out