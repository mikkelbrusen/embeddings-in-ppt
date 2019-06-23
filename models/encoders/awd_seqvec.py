import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import  length_to_mask
from models.utils.awd_model import AWD_Embedding
from models.encoders.deeploc_raw import Encoder as BaseEncoder


class Encoder(BaseEncoder):
  """
  Encoder 

  Inputs: input, seq_len
    - **input** of shape
  Outputs: output
    - **output** of shape 
  """
  def __init__(self, args):
    super().__init__(args)

    self.awd = AWD_Embedding(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True)

    # load pretrained awd
    with open("pretrained_models/awd_lstm/test_v2_statedict.pt", 'rb') as f:
        state_dict = torch.load(f, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    self.awd.load_state_dict(state_dict)

  def forward(self, inp, seq_lengths):
    #### AWD 
    with torch.no_grad():
      all_hid, last_hid, raw_all_hid, dropped_all_hid, emb = self.awd(input=inp, seq_lengths=seq_lengths) 
    
    output = dropped_all_hid[0] + dropped_all_hid[1] #(seq_len, bs, 1280)
    mask = length_to_mask(seq_lengths).unsqueeze(2) # (bs, seq_len, 1)
    output = output.permute(1,0,2) * mask #(bs, seq_len, 1280)
    output = output.sum(1) / mask.sum(1) # (bs, 1280)

    return output