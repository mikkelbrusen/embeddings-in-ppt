import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.awd_model import AWD_Embedding
from models.encoders.deeploc_raw import Encoder as BaseEncoder


class Encoder(nn.Module):
  """
  Encoder that just runs awd

  Inputs: input, seq_len
    - **input** of shape:
  Outputs: output
    - **output** is a tuple: (all_hid, last_hid, raw_all_hid, dropped_all_hid, emb) - NOT batch first
  """
  def __init__(self, args):
    super().__init__()

    self.awd = AWD_Embedding(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True)

    # load pretrained awd
    with open("pretrained_models/awd_lstm/test_v2_statedict.pt", 'rb') as f:
        state_dict = torch.load(f, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    self.awd.load_state_dict(state_dict)

  def forward(self, inp, seq_lengths):

    with torch.no_grad():
      all_hid, last_hid, raw_all_hid, dropped_all_hid, emb = self.awd(input=inp, seq_lengths=seq_lengths)
    
    return all_hid, last_hid, raw_all_hid, dropped_all_hid, emb