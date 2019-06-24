import torch.nn as nn
import torch

from configs.secpred.base import Config as BaseConfig

from models.encoders.elmo import Encoder
from models.decoders.seqvec_secpred import Decoder

class Model(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.encoder = Encoder(args)
    self.decoder = Decoder(args, in_size=320*2)

  def forward(self, inp, seq_len):
    inp = inp.long()
    (elmo_hid, elmo_hid_rev), last_hid, raw_all_hid, dropped_all_hid, emb = self.encoder(inp, seq_len)
    elmo_hid = elmo_hid.permute(1,2,0)
    elmo_hid_rev = elmo_hid_rev.permute(1,2,0)
    inp = torch.cat((elmo_hid, elmo_hid_rev), dim=1)
    output = self.decoder(inp, seq_len)
    return output

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args, Model)