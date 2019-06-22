import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.attention import Attention


class Decoder(nn.Module):
  """
  Decoder structured like DeepLoc

  Inputs: input, seq_len
    - **input** of shape
  Outputs: output
    - **output** of shape 
  """
  def __init__(self, args):
    super().__init__()
    self.args = args
   
    self.relu = nn.ReLU()
    self.drop = nn.Dropout(args.hid_dropout)
    self.attn = Attention(in_size=args.n_hid*2, att_size=args.att_size)

    self.dense = nn.Linear(args.n_hid*2, args.n_hid*2)
    self.label = nn.Linear(args.n_hid*2, args.num_classes)
    self.mem = nn.Linear(args.n_hid*2, 1)
 
    self.init_weights()

  def init_weights(self):
    self.dense.bias.data.zero_()
    torch.nn.init.orthogonal_(self.dense.weight.data, gain=math.sqrt(2))
    
    self.label.bias.data.zero_()
    torch.nn.init.orthogonal_(self.label.weight.data, gain=math.sqrt(2))
    
    for m in self.modules():
      if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
        for name, param in m.named_parameters():
          if 'weight_ih' in name:
            torch.nn.init.orthogonal_(param.data, gain=1)
          elif 'weight_hh' in name:
            torch.nn.init.orthogonal_(param.data, gain=1)
          elif 'bias_ih' in name:
            param.data.zero_()
          elif 'bias_hh' in name:
            param.data.zero_()
      elif type(m) in [nn.Conv1d]:
        for name, param in m.named_parameters():
          if 'weight' in name:
            torch.nn.init.orthogonal_(param.data, gain=math.sqrt(2))           
          if 'bias' in name:
            param.data.zero_()

  def forward(self, inp, seq_lengths):    
    attn_output, alpha = self.attn(x_in=inp, seq_lengths=seq_lengths) #(batch_size, hidden_size*2) alpha = (batch_size, seq_len)
    output = self.drop(attn_output)
    
    output = self.relu(self.dense(output)) # (batch_size, hidden_size*2)
    output = self.drop(output)
    
    out = self.label(output) #(batch_size, num_classes)
    out_mem = torch.sigmoid(self.mem(output)) #(batch_size, 1)
  
    return (out, out_mem), alpha # alpha is only used in notebooks