import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import init_weights
from models.utils.awd_model import AWDEmbedding

class Encoder(nn.Module):
  """
  Encoder with bi_awd concatenated to the LSTM output

  Parameters:
    -- awd_layer: last or second
    -- project_size: size of projection layer from bi_awd to lstm
second
  Inputs: input, seq_len
    - **input** of shape
  Outputs: output
    - **output** of shape (batch_size, seq_len, hidden_size*2)
  """
  def __init__(self, args, awd_layer, project_size=None):
    super().__init__()
    self.args = args
    self.awd_layer = awd_layer
    self.project_size = project_size
    self.drop = nn.Dropout(args.hid_dropout)

    if project_size is not None and awd_layer in ["second"]:
      self.project = nn.Linear(1280, project_size, bias=False)
    elif project_size is not None and awd_layer in ["last"]:
      self.project = nn.Linear(320, project_size, bias=False)

    if project_size is not None:
      self.lstm = nn.LSTM(project_size, args.n_hid, bidirectional=True, batch_first=True)
    elif awd_layer in ["second"]:
      self.lstm = nn.LSTM(1280, args.n_hid, bidirectional=True, batch_first=True)
    elif awd_layer in ["last"]:
      self.lstm = nn.LSTM(320, args.n_hid, bidirectional=True, batch_first=True)

    init_weights(self)
        
    self.awd = AWDEmbedding(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True)
    self.awd.load_pretrained()

  def forward(self, inp, seq_lengths):
    #### AWD 
    with torch.no_grad():
      all_hid, _, _ = self.awd(input=inp, seq_lengths=seq_lengths)

    if self.awd_layer == "last":
      awd_hid = all_hid[2]
      awd_hid = awd_hid.permute(1,0,2) # (bs, seq_len, 320)

    elif self.awd_layer == "second":
      awd_hid = all_hid[1]
      awd_hid = awd_hid.permute(1,0,2) # (bs, seq_len, 1280) 

    if self.project_size is not None:
      awd_hid = self.project(awd_hid) # (bs, seq_len, project_size) 
    ### End AWD

    awd_hid = self.drop(awd_hid) #( batch_size, seq_len, project_size or 1280)
    
    pack = nn.utils.rnn.pack_padded_sequence(awd_hid, seq_lengths, batch_first=True)
    packed_output, _ = self.lstm(pack) #h = (2, batch_size, hidden_size)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) #(batch_size, seq_len, hidden_size*2)
  
    return output