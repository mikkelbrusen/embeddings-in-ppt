import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.utils import init_weights

class Decoder(nn.Module):

  def __init__(self, args, in_size):
    super().__init__()
    self.args = args
    
    self.drop = nn.Dropout(p=0.25)
    self.relu = nn.ReLU()

    self.cnn_1 = nn.Conv1d(in_channels=in_size, out_channels=32, kernel_size=7, padding= 7//2)
    self.cnn_2 = nn.Conv1d(in_channels=32, out_channels=8, kernel_size=7, padding= 7//2)

    init_weights(self)
    
  def forward(self, inp, seq_lengths):
    output = self.drop(self.relu(self.cnn_1(inp)))
    out = self.cnn_2(output).permute(0,2,1)
  
    return out