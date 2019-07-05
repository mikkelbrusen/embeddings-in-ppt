import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import rename_state_dict_keys
from models.utils.bi_awd_model import BiAWDEmbedding, key_transformation
from models.encoders.deeploc_raw import Encoder as BaseEncoder


class Encoder(nn.Module):
  """
  Encoder that just runs awd

  Inputs: input, seq_len
    - **input** of shape:
  Outputs: output
    - **output** is a tuple: (all_hid, last_hidden_states, emb) - NOT batch first
  """
  def __init__(self, args):
    super().__init__()

    self.bi_awd = BiAWDEmbedding(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True)
    self.bi_awd.load_pretrained()

  def forward(self, inp, seq_lengths):

    with torch.no_grad():
        (outputs, outputs_rev), (hidden, hidden_rev), emb = self.bi_awd(input=inp, seq_lengths=seq_lengths)
    
    return (outputs, outputs_rev), (hidden, hidden_rev), emb