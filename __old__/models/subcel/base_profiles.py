import math
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F

from models.utils.attention import Attention, MultiStepAttention

from models.subcel.base import Model as BaseModel
from models.subcel.base import Config as BaseConfig

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args)
    self.Model = Model

  def _prepare_tensors(self, batch):
    inputs, targets, in_masks, targets_mem, unk_mem = batch

    seq_lengths = in_masks.sum(1).astype(np.int32)

    #sort to be in decending order for pad packed to work
    perm_idx = np.argsort(-seq_lengths)
    in_masks = in_masks[perm_idx]
    seq_lengths = seq_lengths[perm_idx]
    inputs = inputs[perm_idx]
    targets = targets[perm_idx]
    targets_mem = targets_mem[perm_idx]
    unk_mem = unk_mem[perm_idx]

    #convert to tensors
    inputs = torch.from_numpy(inputs).to(self.args.device) # (batch_size, seq_len, n_feat)
    in_masks = torch.from_numpy(in_masks).to(self.args.device)
    seq_lengths = torch.from_numpy(seq_lengths).to(self.args.device)

    return inputs, seq_lengths, in_masks, targets, targets_mem, unk_mem

class Model(BaseModel):
  def __init__(self, args):
    super().__init__(args)
    self.args = args
    self.in_drop1d = nn.Dropout(args.in_dropout1d)
    self.in_drop2d = nn.Dropout2d(args.in_dropout2d)
    self.drop = nn.Dropout(args.hid_dropout)

    self.convs = nn.ModuleList([nn.Conv1d(in_channels=args.n_features, out_channels=args.n_filters, kernel_size=i, padding=i//2) for i in args.conv_kernels])
    self.cnn_final = nn.Conv1d(in_channels=len(self.convs)*args.n_filters, out_channels=128, kernel_size=3, padding= 3//2)
    self.relu = nn.ReLU()

    self.lstm = nn.LSTM(128, args.n_hid, bidirectional=True, batch_first=True)
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
    
  def forward(self, inp, seq_lengths): # inp: (batch_size, seq_len, n_feat)
    inp = self.in_drop1d(inp) # feature dropout
    x = self.in_drop2d(inp)  # (batch_size, seq_len, n_feat) - 2d dropout

    x = x.permute(0, 2, 1)  # (batch_size, n_feat, seq_len)
    conv_cat = torch.cat([self.relu(conv(x)) for conv in self.convs], dim=1) # (batch_size, n_feat*len(convs), seq_len)
    x = self.relu(self.cnn_final(conv_cat)) #(batch_size, out_channels=128, seq_len)

    x = x.permute(0, 2, 1) #(batch_size, seq_len, out_channels=128)
    x = self.drop(x) #( batch_size, seq_len, lstm_input_size)
    
    pack = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
    packed_output, (h, c) = self.lstm(pack) #h = (2, batch_size, hidden_size)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) #(batch_size, seq_len, hidden_size*2)
  
    attn_output, alpha = self.attn(x_in=output, seq_lengths=seq_lengths) #(batch_size, hidden_size*2) alpha = (batch_size, seq_len)
    output = self.drop(attn_output)
    
    output = self.relu(self.dense(output)) # (batch_size, hidden_size*2)
    output = self.drop(output)
    
    out = self.label(output) #(batch_size, num_classes)
    out_mem = torch.sigmoid(self.mem(output)) #(batch_size, 1)

    return (out, out_mem), alpha # alpha only used for visualization in notebooks