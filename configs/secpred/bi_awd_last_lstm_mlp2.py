import torch.nn as nn

from configs.secpred.base import Config as BaseConfig

from models.encoders.bi_awd import Encoder
from models.decoders.lstm_mlp2 import Decoder

import torch

class Model(nn.Module):
  def __init__(self, args):
    super().__init__()

    self.args = args
    self.encoder = Encoder(args)
    self.decoder = Decoder(args, in_size=640)

  def forward(self, inp, seq_len):
    inp = inp.long()
    (all_hid, all_hid_rev) , _, _ = self.encoder(inp, seq_len)
    elmo_hid = all_hid[2]
    elmo_hid_rev = all_hid_rev[2]

    elmo_hid = elmo_hid.permute(1,0,2) # (bs, seq_len, 1280) 
    elmo_hid_rev = elmo_hid_rev.permute(1,0,2) # (bs, seq_len, 1280) 
    
    elmo_hid = torch.cat((elmo_hid, elmo_hid_rev), dim=2) # (bs, seq_len, something big) 
    del elmo_hid_rev
    ### End BiAWDEmbedding 
    output = self.decoder(elmo_hid, seq_len)
    return output

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args, Model)