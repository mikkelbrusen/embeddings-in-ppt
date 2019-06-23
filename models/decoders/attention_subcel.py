import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import init_weights
from models.utils.attention import Attention


class Decoder(nn.Module):
  """
  Decoder 

  Inputs: input, seq_len
    - **input** of shape
  Outputs: output
    - **output** of shape 
  """
  def __init__(self, args, in_size):
    super().__init__()
    self.args = args
   
    self.attn = Attention(in_size=in_size, att_size=args.att_size)
    self.in_drop = nn.Dropout2d(args.in_dropout1d)
    self.label = nn.Linear(in_size, args.num_classes)
    self.mem = nn.Linear(in_size, 1)

    init_weights(self)

  def forward(self, inp, seq_lengths):    
    output = self.in_drop(inp) #(batch_size, seq_len, in_size)
    
    attn_output, alpha = self.attn(x_in=output, seq_lengths=seq_lengths) #(batch_size, in_size)

    out = self.label(attn_output) #(batch_size, num_classes)
    out_mem = torch.sigmoid(self.mem(attn_output)) #(batch_size, 1)

    return (out, out_mem), alpha # alpha only used for visualization in notebooks