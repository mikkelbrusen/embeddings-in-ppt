import torch.nn as nn

from configs.secpred.base import Config as BaseConfig

from models.encoders.awd import Encoder
from models.decoders.lstm_mlp2 import Decoder

import torch

class Model(nn.Module):
  def __init__(self, args):
    super().__init__()

    self.args = args
    self.encoder = Encoder(args)
    self.decoder = Decoder(args, in_size=1280)

  def forward(self, inp, seq_len):
    inp = inp.long()
    all_hid, _, _  = self.encoder(inp, seq_len)


    awd_hid = all_hid[0]
    awd_hid = awd_hid.permute(1,0,2) # (bs, seq_len, 1280) 

    output = self.decoder(awd_hid, seq_len)
    return output

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args, Model)