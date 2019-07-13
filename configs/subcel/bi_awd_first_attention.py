import torch.nn as nn
import torch

from configs.subcel.base import Config as BaseConfig

from models.encoders.bi_awd import Encoder
from models.decoders.attention_subcel import Decoder

class Model(nn.Module):
  def __init__(self, args):
    super().__init__()

    self.args = args
    self.encoder = Encoder(args)
    self.decoder = Decoder(args, in_size=1280*2)

  def forward(self, inp, seq_len):
    (all_hid, all_hid_rev), _, _ = self.encoder(inp, seq_len)

    #perform a permutation of the output since the decoder needs it batch first
    output = torch.cat((all_hid[0], all_hid_rev[0]), dim=2) # (seq_len, bs, 2560)
    del all_hid, all_hid_rev
    output = output.permute(1,0,2) # (bs, seq_len, 2560)

    output = self.decoder(output, seq_len)
    return output

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args, Model)