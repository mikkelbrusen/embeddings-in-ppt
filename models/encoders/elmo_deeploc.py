import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.elmo import Encoder as Elmo
from models.encoders.deeploc_raw import Encoder as BaseEncoder


class Encoder(BaseEncoder):
  """
  Encoder with elmo concatenated to the LSTM output

  Parameters:
    -- architecture: before, after or both
    -- elmo_layer: last, 2ndlast, h, c or hc
      where h c and hc are from the 2nd last layer in elmo

  Inputs: input, seq_len
    - **input** of shape
  Outputs: output
    - **output** of shape (batch_size, seq_len, hidden_size*2 + 300) if arch is after/both else 
      (batch_size, seq_len, hidden_size*2)
  """
  def __init__(self, args, architecture, elmo_layer):
    super().__init__(args)
    self.architecture = architecture
    self.elmo_layer = elmo_layer

    self.elmo = Elmo(args)
    if elmo_layer in ["h","c","2ndlast"]:
      self.project = nn.Linear(2560, 300)
    elif elmo_layer in ["last"]:
      self.project = nn.Linear(320*2, 300)
    elif elmo_layer in ["hc"]:
      self.project = nn.Linear(5120, 300)

    if self.architecture in ["before", "both"]:
      self.lstm = nn.LSTM(128+300, args.n_hid, bidirectional=True, batch_first=True)

  def forward(self, inp, seq_lengths):
    all_hid, last_hid, raw_all_hid, dropped_all_hid, emb = self.elmo(inp, seq_lengths)
    
    if self.elmo_layer == "h":
      elmo_hid, elmo_hid_rev = last_hid

      elmo_hid = elmo_hid[1][0].permute(1,0,2) # (bs, seq_len, 1280) 
      elmo_hid_rev = elmo_hid_rev[1][0].permute(1,0,2) # (bs, seq_len, 1280) 

    elif self.elmo_layer == "c":
      elmo_hid, elmo_hid_rev = last_hid

      elmo_hid = elmo_hid[1][1].permute(1,0,2) # (bs, seq_len, 1280) 
      elmo_hid_rev = elmo_hid_rev[1][1].permute(1,0,2) # (bs, seq_len, 1280) 

    elif self.elmo_layer == "hc":
      elmo_hid, elmo_hid_rev = last_hid

      elmo_hid_h = elmo_hid[1][0].permute(1,0,2) # (bs, seq_len, 1280) 
      elmo_hid_rev_h = elmo_hid_rev[1][0].permute(1,0,2) # (bs, seq_len, 1280) 

      elmo_hid_c = elmo_hid[1][1].permute(1,0,2) # (bs, seq_len, 1280) 
      elmo_hid_rev_c = elmo_hid_rev[1][1].permute(1,0,2) # (bs, seq_len, 1280) 

      elmo_hid = torch.cat((elmo_hid_h, elmo_hid_c), dim=2) # (bs, seq_len, 2560)
      elmo_hid_rev = torch.cat((elmo_hid_rev_h, elmo_hid_rev_c), dim=2) # (bs, seq_len, 2560)
    
    elif self.elmo_layer == "last":
      elmo_hid, elmo_hid_rev = all_hid

      elmo_hid = elmo_hid.permute(1,0,2) # (bs, seq_len, 320) 
      elmo_hid_rev = elmo_hid_rev.permute(1,0,2) # (bs, seq_len, 320) 

    elif self.elmo_layer == "2ndlast":
      elmo_hid, elmo_hid_rev = dropped_all_hid

      elmo_hid = elmo_hid[1].permute(1,0,2) # (bs, seq_len, 1280) 
      elmo_hid_rev = elmo_hid_rev[1].permute(1,0,2) # (bs, seq_len, 1280) 
    
    elmo_hid = torch.cat((elmo_hid, elmo_hid_rev), dim=2) # (bs, seq_len, something big) 


    elmo_hid = self.project(elmo_hid) # (bs, seq_len, 300) 
    ### End Elmo 
    
    inp = self.embed(inp) # (batch_size, seq_len, emb_size)

    inp = self.in_drop1d(inp) # feature dropout
    x = self.in_drop2d(inp)  # (batch_size, seq_len, emb_size) - 2d dropout

    x = x.permute(0, 2, 1)  # (batch_size, emb_size, seq_len)
    conv_cat = torch.cat([self.relu(conv(x)) for conv in self.convs], dim=1) # (batch_size, emb_size*len(convs), seq_len)
    x = self.relu(self.cnn_final(conv_cat)) #(batch_size, out_channels=128, seq_len)

    x = x.permute(0, 2, 1) #(batch_size, seq_len, out_channels=128)
    if self.architecture in ["before", "both"]:
      x = torch.cat((x, elmo_hid), dim=2)
    x = self.drop(x) #( batch_size, seq_len, lstm_input_size)
    
    pack = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
    packed_output, (h, c) = self.lstm(pack) #h = (2, batch_size, hidden_size)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) #(batch_size, seq_len, hidden_size*2)

    if self.architecture in ["after", "both"]:
      output = torch.cat((output, elmo_hid), dim=2) # (batch_size, seq_len, hidden_size*2+300)
  
    return output