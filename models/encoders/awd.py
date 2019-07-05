import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.awd_model import AWDEmbedding

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

    self.awd = AWDEmbedding(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True)
    self.awd.load_pretrained()

  def forward(self, inp, seq_lengths):

    with torch.no_grad():
      outputs, hidden, emb = self.awd(input=inp, seq_lengths=seq_lengths)
    
    return outputs, hidden, emb