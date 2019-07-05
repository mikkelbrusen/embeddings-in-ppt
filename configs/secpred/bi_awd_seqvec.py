import torch.nn as nn
import torch

from configs.secpred.base import Config as BaseConfig

from models.encoders.bi_awd import Encoder
from models.decoders.seqvec_secpred import Decoder

class Model(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.encoder = Encoder(args)
    self.decoder = Decoder(args, in_size=2560)

  def forward(self, inp, seq_len):
    inp = inp.long()
    (elmo_hid, elmo_hid_rev), _, _ = self.encoder(inp, seq_len)

    # Add backward and forward
    output = torch.cat(((elmo_hid[0] + elmo_hid[1]), (elmo_hid_rev[0] + elmo_hid_rev[1])), dim=2) #(seq_len, bs, 2560)
    del elmo_hid, elmo_hid_rev
    inp = output.permute(1,2,0)
    output = self.decoder(inp, seq_len)
    return output

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args, Model)