import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import init_weights
from models.utils.attention import Attention


class Decoder(nn.Module):
  """
  Decoder structured seqvec for subcellular localization

  Inputs: input, seq_len
    - **input** of shape
  Outputs: output
    - **output** of shape 
  """
  def __init__(self, args, in_size):
    super().__init__()
    self.args = args
   
    self.linear = nn.Linear(in_size, 32)
    self.drop = nn.Dropout(0.25)
    self.relu = nn.ReLU()
    self.bn = nn.BatchNorm1d(32)
    self.label = nn.Linear(32, args.num_classes)
    self.mem = nn.Linear(32, 1)

    init_weights(self)
    

  def forward(self, inp):    
    output = self.bn(self.relu(self.drop(self.linear(inp))))

    out = self.label(output) #(batch_size, num_classes)
    out_mem = torch.sigmoid(self.mem(output)) #(batch_size, 1)

    return (out, out_mem), None # alpha only used for visualization in notebooks