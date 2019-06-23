import torch.nn as nn
import torch

from configs.subcel.base import Config as BaseConfig

from models.encoders.elmo import Encoder
from models.decoders.attention_subcel import Decoder

class Model(nn.Module):
  def __init__(self, args):
    super().__init__()

    self.args = args
    self.encoder = Encoder(args)
    self.decoder = Decoder(args, in_size=320*2)

  def forward(self, inp, seq_len):
    all_hid, last_hid, raw_all_hid, dropped_all_hid, emb = self.encoder(inp, seq_len)

    (all_hid, all_hid_rev) = all_hid
    #perform a permutation of the output since the decoder needs it batch first
    decoder_input = torch.cat((all_hid, all_hid_rev), dim=2) # (seq_len, bs, in_size)
    decoder_input = decoder_input.permute(1,0,2) # (bs, seq_len, in_size)

    output = self.decoder(decoder_input, seq_len)
    return output

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args, Model)