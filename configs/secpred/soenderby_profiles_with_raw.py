import torch.nn as nn

from configs.secpred.base import Config as BaseConfig
from utils.utils import get_raw_from_one_hot
from models.encoders.bi_awd_soenderby_profiles import Encoder
from models.decoders.mlp2 import Decoder
import torch

class Model(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.encoder = Encoder(args=args, bi_awd_layer="last", architecture="both")
    self.decoder = Decoder(args, in_size=self.args.n_hid*2+300)

  def forward(self, inp, seq_len):
    raw = get_raw_from_one_hot(inp[:,:,torch.arange(0,22)])
    a = torch.arange(0,21)
    b = torch.arange(22,43)
    c = torch.cat((a,b))
    inp = inp[:,:,c]
    output = self.encoder(inp, raw, seq_len)
    output = self.decoder(output, seq_len)
    return output

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args, Model)