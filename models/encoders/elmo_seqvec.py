import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import rename_state_dict_keys, length_to_mask
from models.encoders.deeploc_raw import Encoder as BaseEncoder
from models.encoders.elmo import Encoder as Elmo


class Encoder(nn.Module):
  """
  Encoder 

  Inputs: input, seq_len
    - **input** of shape
  Outputs: output
    - **output** of shape 
  """
  def __init__(self, args):
    super().__init__(args)

    self.elmo = Elmo(args)

  def forward(self, inp, seq_lengths):
    all_hid, last_hid, raw_all_hid, dropped_all_hid, emb = self.elmo(inp, seq_lengths)
    
    (elmo_hid, elmo_hid_rev) = dropped_all_hid # [(seq_len, bs, 1280),(seq_len, bs, 1280),(seq_len, bs, emb_size)] , ...
    output = torch.cat(((elmo_hid[0] + elmo_hid[1]), (elmo_hid_rev[0] + elmo_hid_rev[1])), dim=2) #(seq_len, bs, 2560)
    mask = length_to_mask(seq_lengths).unsqueeze(2) # (bs, seq_len, 1)
    output = output.permute(1,0,2) * mask #(bs, seq_len, 2560)
    output = output.sum(1) / mask.sum(1) # (bs, 2560)

    return output