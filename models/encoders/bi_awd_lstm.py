import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import rename_state_dict_keys, init_weights
from models.utils.bi_awd_model import BiAWDEmbedding, key_transformation
from models.encoders.deeploc_raw import Encoder as BaseEncoder


class Encoder(nn.Module):
  """
  Encoder with elmo concatenated to the LSTM output

  Parameters:
    -- project_size: size of projection layer from elmo to lstm

  Inputs: input, seq_len
    - **input** of shape
  Outputs: output
    - **output** of shape (batch_size, seq_len, hidden_size*2)
  """
  def __init__(self, args, project_size=None):
    super().__init__()
    self.args = args
    self.project_size = project_size
    self.drop = nn.Dropout(args.hid_dropout)

    if project_size is not None:
      self.project = nn.Linear(2*1280, project_size, bias=False)

    self.lstm = nn.LSTM(project_size if project_size is not None else 2*1280, args.n_hid, bidirectional=True, batch_first=True)
    
    init_weights(self)
    
    self.bi_awd = BiAWDEmbedding(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True)
    self.bi_awd.load_pretrained()

  def forward(self, inp, seq_lengths):
    with torch.no_grad():
        (all_hid, all_hid_rev) , _, _ = self.bi_awd(inp, seq_lengths) # all_hid, last_hidden_states, emb
    
    elmo_hid = all_hid[1].permute(1,0,2) # (bs, seq_len, 1280) 
    elmo_hid_rev = all_hid_rev[1].permute(1,0,2) # (bs, seq_len, 1280) 
    
    elmo_hid = torch.cat((elmo_hid, elmo_hid_rev), dim=2) # (bs, seq_len, 1280) 

    if self.project_size is not None:
      elmo_hid = self.project(elmo_hid) # (bs, seq_len, project_size) 
    del elmo_hid_rev
    ### End BiAWDEmbedding 

    elmo_hid = self.drop(elmo_hid) #( batch_size, seq_len, project_size or 2*1280)
    
    pack = nn.utils.rnn.pack_padded_sequence(elmo_hid, seq_lengths, batch_first=True)
    packed_output, _ = self.lstm(pack) #h = (2, batch_size, hidden_size)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) #(batch_size, seq_len, hidden_size*2)
  
    return output