import torch
import torch.nn as nn

from utils.utils import length_to_mask
from configs.subcel.base import Config as BaseConfig

from models.encoders.bi_awd import Encoder
from models.decoders.seqvec_subcel import Decoder

class Model(nn.Module):
  def __init__(self, args):
    super().__init__()

    self.args = args
    self.encoder = Encoder(args)
    self.decoder = Decoder(args, in_size=2560)

  def forward(self, inp, seq_len):
    (bi_awd_hid, bi_awd_hid_rev), _, _ = self.encoder(inp, seq_len)

    # Add backward and forward, and mean over sequence
    output = torch.cat(((bi_awd_hid[0] + bi_awd_hid[1]), (bi_awd_hid_rev[0] + bi_awd_hid_rev[1])), dim=2) #(seq_len, bs, 2560)
    del bi_awd_hid, bi_awd_hid_rev
    mask = length_to_mask(seq_len).unsqueeze(2) # (bs, seq_len, 1)
    output = output.permute(1,0,2) * mask #(bs, seq_len, 2560)
    output = output.sum(1) / mask.sum(1) # (bs, 2560)


    output = self.decoder(output)
    return output

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args, Model)