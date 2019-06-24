import torch.nn as nn

from configs.subcel.base import Config as BaseConfig

from models.encoders.elmo_deeploc import Encoder
from models.decoders.deeploc_subcel import Decoder

class Model(nn.Module):
  def __init__(self, args):
    super().__init__()

    self.args = args
    self.encoder = Encoder(args, architecture="after", elmo_layer="last")
    self.decoder = Decoder(args, in_size=args.n_hid*2+300)

  def forward(self, inp, seq_len):
    output = self.encoder(inp, seq_len)
    output = self.decoder(output, seq_len)
    return output

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args, Model)