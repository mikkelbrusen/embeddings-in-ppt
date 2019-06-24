import torch.nn as nn

from configs.subcel.base import Config as BaseConfig

from models.encoders.awd_seqvec import Encoder
from models.decoders.seqvec_subcel import Decoder

class Model(nn.Module):
  def __init__(self, args):
    super().__init__()

    self.args = args
    self.encoder = Encoder(args)
    self.decoder = Decoder(args, in_size=1280)

  def forward(self, inp, seq_len):
    output = self.encoder(inp, seq_len)
    output = self.decoder(output)
    return output

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args, Model)