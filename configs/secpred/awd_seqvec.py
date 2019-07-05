import torch.nn as nn
import torch

from configs.secpred.base import Config as BaseConfig

from models.encoders.awd import Encoder
from models.decoders.seqvec_secpred import Decoder

class Model(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.encoder = Encoder(args)
    self.decoder = Decoder(args, in_size=1280)

  def forward(self, inp, seq_len):
    inp = inp.long()
    output, _, _ = self.encoder(inp, seq_len)

    # Add the add the layers and mean over sequence
    output = output[0] + output[1] #(seq_len, bs, 1280)
    inp = output.permute(1,2,0)
    output = self.decoder(inp, seq_len)
    return output

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args, Model)