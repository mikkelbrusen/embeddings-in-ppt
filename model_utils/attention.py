import math
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0,'..')
from utils import length_to_negative_mask

class Attention(nn.Module):
  def __init__(self, in_size, att_size):
    super(Attention, self).__init__()
    self.linear_in = nn.Linear(in_size, att_size)
    self.linear_att = nn.Linear(att_size, 1, bias=False)
    
  def forward(self, x_in, seq_lengths):  # x_in.shape: [bs, seq_len, in_size]
    att_vector = self.linear_in(x_in) # [bs, seq_len, att_size]
    att_hid_align = torch.tanh(att_vector) # [bs, seq_len, att_size]
    att_score = self.linear_att(att_hid_align).squeeze(2) # [bs, seq_len]
    
    mask = length_to_negative_mask(seq_lengths)
    alpha = F.softmax(att_score + mask, dim=1) # [bs, seq_len]
    att = alpha.unsqueeze(2) # [bs, seq_len, 1]

    return torch.sum(x_in * att, dim=1), alpha # [bs, in_size]


class MultiStepAttention(nn.Module):
  def __init__(self, input_size, hidden_size, att_size=256, cell_hid_size=512, num_steps=10, directions=2):
    super(MultiStepAttention, self).__init__()
    self.directions = directions
    self.num_steps = num_steps

    # Attention
    self.linear_in = nn.Linear(input_size, att_size)
    self.linear_hid = nn.Linear(cell_hid_size, att_size)
    self.linear_att = nn.Linear(att_size, 1, bias=False)
    
    self.use_projection = hidden_size*self.directions != cell_hid_size
    if (self.use_projection):
      self.linear_proj = nn.Linear(hidden_size*self.directions, cell_hid_size)
      
    self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=cell_hid_size)

  def forward(self, x_in, hidden, seq_lengths): #x_in: (batch_size, seq_len, hidden_size*directions)
    h_x, c_x = hidden #h_x: (directions, batch_size, hidden_size)

    # Prepare hidden states for lstm_cell
    h_x = h_x.permute(1,0,2) # (batch_size, directions, hidden_size)
    c_x = c_x.permute(1,0,2)
    h_x = h_x.reshape(h_x.size(0), h_x.size(2)*self.directions) #(batch_size, hidden_size*directions)
    c_x = c_x.reshape(c_x.size(0), c_x.size(2)*self.directions)

    if (self.use_projection):
      h_x = self.linear_proj(h_x) # (batch_size, cell_hid_size)
      c_x = self.linear_proj(c_x) # (batch_size, cell_hid_size)

    for i in range(self.num_steps):
      # Attention
      att_vector = self.linear_in(x_in) # [bs, seq_len, att_size]
      hid_vector = self.linear_hid(h_x) # [bs, att_size]
      att_vector = att_vector + hid_vector.unsqueeze(dim=1)
      att_hid_align = torch.tanh(att_vector) # [bs, seq_len, att_size]
      att_score = self.linear_att(att_hid_align).squeeze(2) # [bs, seq_len]
      mask = length_to_negative_mask(seq_lengths)

      alpha = F.softmax(att_score + mask, dim=1) # [bs, seq_len]
      att = alpha.unsqueeze(2) # [bs, seq_len, 1]

      c_vector = torch.sum(x_in * att, dim=1) # (batch_size, hidden_size*2)
      # /attention

      h_x, c_x = self.lstm_cell(input=c_vector, hx=(h_x,c_x)) # h_x: (batch_size, hidden_size*2)

    # output c_vector rather tan h_x to reduce parameters and overfitting
    return c_vector, alpha