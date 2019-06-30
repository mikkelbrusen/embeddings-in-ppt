import torch.nn as nn

from utils.utils import length_to_mask
from configs.subcel.base import Config as BaseConfig

from models.encoders.awd import Encoder
from models.decoders.seqvec_subcel import Decoder

class Model(nn.Module):
  def __init__(self, args):
    super().__init__()

    self.args = args
    self.encoder = Encoder(args)
    self.decoder = Decoder(args, in_size=1280)

  def forward(self, inp, seq_len):
    output, _, _ = self.encoder(inp, seq_len)

    # Add the add the layers and mean over sequence
    output = output[0] + output[1] #(seq_len, bs, 1280)
    mask = length_to_mask(seq_len).unsqueeze(2) # (bs, seq_len, 1)
    output = output.permute(1,0,2) * mask #(bs, seq_len, 1280)
    output = output.sum(1) / mask.sum(1) # (bs, 1280)

    output = self.decoder(output)
    return output

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args, Model)