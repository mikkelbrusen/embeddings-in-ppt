import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.elmo_bi import Elmo, key_transformation
from utils.utils import init_weights


class Encoder(nn.Module):
  """
  Encoder structured like DeepLoc

  Inputs: input, seq_len
    - **input** of shape
  Outputs: output
    - **output** of shape (batch_size, seq_len, hidden_size*2)
  """
  def __init__(self, args):
    super().__init__()

    self.args = args
    self.in_drop1d = nn.Dropout(args.in_dropout1d)
    self.in_drop2d = nn.Dropout2d(args.in_dropout2d)
    self.drop = nn.Dropout(args.hid_dropout)

    self.embed = nn.Embedding(num_embeddings=21, embedding_dim=args.n_features, padding_idx=20)

    self.convs = nn.ModuleList([nn.Conv1d(in_channels=args.n_features, out_channels=args.n_filters, kernel_size=i, padding=i//2) for i in args.conv_kernels])
    self.cnn_final = nn.Conv1d(in_channels=len(self.convs)*args.n_filters, out_channels=128, kernel_size=3, padding= 3//2)
    self.relu = nn.ReLU()

    self.lstm = nn.LSTM(128, args.n_hid, bidirectional=True, batch_first=True)

    init_weights(self)
    

  def forward(self, inp, seq_lengths):    
    inp = self.embed(inp) # (batch_size, seq_len, emb_size)

    inp = self.in_drop1d(inp) # feature dropout
    x = self.in_drop2d(inp)  # (batch_size, seq_len, emb_size) - 2d dropout

    x = x.permute(0, 2, 1)  # (batch_size, emb_size, seq_len)
    conv_cat = torch.cat([self.relu(conv(x)) for conv in self.convs], dim=1) # (batch_size, emb_size*len(convs), seq_len)
    x = self.relu(self.cnn_final(conv_cat)) #(batch_size, out_channels=128, seq_len)

    x = x.permute(0, 2, 1) #(batch_size, seq_len, out_channels=128)
    x = self.drop(x) #( batch_size, seq_len, lstm_input_size)
    
    pack = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
    packed_output, (h, c) = self.lstm(pack) #h = (2, batch_size, hidden_size)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) #(batch_size, seq_len, hidden_size*2)
  
    return output