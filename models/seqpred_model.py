import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import crf

class SeqPred(nn.Module):
  def __init__(self, input_size=42, num_units_encoder=400, num_units_l1=200, num_units_l2=200, number_outputs=8):
    super(SeqPred, self).__init__()

    self.number_outputs = number_outputs
    self.densel1 = nn.Linear(input_size, num_units_l1, bias=False)
    self.bn1 = nn.BatchNorm1d(num_features=num_units_l1)
    self.gru = nn.GRU(num_units_l1+input_size, num_units_encoder, bidirectional=True, batch_first=True)
    self.drop = nn.Dropout()
    self.relu = nn.ReLU()
    
    self.densel2 = nn.Linear(num_units_encoder*2, num_units_l2)
    self.densel3 = nn.Linear(num_units_l2, number_outputs**2)
    self.label = nn.Linear(num_units_l2, number_outputs)
 
    self.init_weights()
    
    
  def init_weights(self):
    #self.densel1.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.densel1.weight.data, gain=1)
    
    self.densel2.bias.data.zero_()
    torch.nn.init.xavier_uniform_(self.densel2.weight.data, gain=1)

    self.densel3.bias.data.zero_()
    torch.nn.init.xavier_uniform_(self.densel3.weight.data, gain=1)

    self.label.bias.data.zero_()
    torch.nn.init.xavier_uniform_(self.label.weight.data, gain=1)
    
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
    
  def forward(self, inp, seq_lengths, mask):
    x = self.densel1(inp).permute(0,2,1)
    x = self.bn1(x).permute(0,2,1)
    x = self.relu(x)
    x = torch.cat((inp,x), dim=2)
    pack = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
    packed_output, _ = self.gru(pack)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
    
    output = self.drop(output)

    output = self.relu(self.densel2(output))
    l_2 = self.densel3(output)
    g = l_2.view(output.size(0), output.size(1), self.number_outputs, self.number_outputs).permute(1,0,2,3)
    f = self.label(output).permute(1,0,2)
    nu_alp = crf.forward_pass(f=f, g=g, mask=mask)
    nu_bet = crf.backward_pass(f=f, g=g, mask=mask)
    return f, g, nu_alp, nu_bet

