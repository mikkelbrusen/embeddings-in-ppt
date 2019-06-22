import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.deeploc_raw import Encoder as BaseEncoder
from models.utils.elmo_bi import Elmo, key_transformation


class Encoder(BaseEncoder):
  """
  Encoder structured like DeepLoc with profiles as input

  Inputs: input, seq_len
    - **input** of shape
  Outputs: output
    - **output** of shape (batch_size, seq_len, hidden_size*2)
  """
  def __init__(self, args):
    super().__init__(args)

  def forward(self, inp, seq_lengths): # inp: (batch_size, seq_len, n_feat)
    inp = self.in_drop1d(inp) # feature dropout
    x = self.in_drop2d(inp)  # (batch_size, seq_len, n_feat) - 2d dropout

    x = x.permute(0, 2, 1)  # (batch_size, n_feat, seq_len)
    conv_cat = torch.cat([self.relu(conv(x)) for conv in self.convs], dim=1) # (batch_size, n_feat*len(convs), seq_len)
    x = self.relu(self.cnn_final(conv_cat)) #(batch_size, out_channels=128, seq_len)

    x = x.permute(0, 2, 1) #(batch_size, seq_len, out_channels=128)
    x = self.drop(x) #( batch_size, seq_len, lstm_input_size)
    
    pack = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
    packed_output, (h, c) = self.lstm(pack) #h = (2, batch_size, hidden_size)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) #(batch_size, seq_len, hidden_size*2)
  
    return output