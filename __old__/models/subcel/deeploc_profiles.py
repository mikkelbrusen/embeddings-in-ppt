import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.subcel.base_profiles import Model as BaseModel
from models.subcel.base_profiles import Config as BaseConfig

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args)
    self.Model = Model

class Model(BaseModel):
  def __init__(self, args):
    super().__init__(args)