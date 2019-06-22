import math
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F

from models.utils.attention import Attention, MultiStepAttention

from configs.subcel.base import Config as BaseConfig

from models.encoders.deeploc_raw import Encoder
from models.decoders.deeploc import Decoder

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args, Encoder, Decoder)