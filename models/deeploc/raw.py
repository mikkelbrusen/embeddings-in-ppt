import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.deeploc.base import Model as BaseModel

class Model(BaseModel):
  def __init__(self, args):
    super().__init__(args)