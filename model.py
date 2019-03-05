import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import length_to_mask

class Attention(nn.Module):
  def __init__(self, in_size, hid_size, att_size):
    super(Attention, self).__init__()
    self.linear_in = nn.Linear(in_size, att_size)
    self.linear_hid = nn.Linear(hid_size, att_size)
    self.linear_att = nn.Linear(att_size, 1, bias=False)
    
  def forward(self, x_in, h_prev, seq_lengths):  # x_in.shape: [bs, seq_len, in_size]
    att_vector = self.linear_in(x_in) # [bs, seq_len, att_size]
    hid_vector = self.linear_hid(h_prev) # [bs, att_size]

    att_hid_align = torch.tanh(att_vector + hid_vector.unsqueeze(dim=1)) # [bs, seq_len, att_size]
   
    att_score = self.linear_att(att_hid_align).squeeze(2) # [bs, seq_len]
    
    mask = length_to_mask(seq_lengths)

    alpha = F.softmax(att_score + mask, dim=1) # [bs, seq_len]
    att = alpha.unsqueeze(2) # [bs, seq_len, 1]

    return torch.sum(x_in * att, dim=1), alpha # [bs, in_size]

class MultiStepAttention(nn.Module):
  def __init__(self, input_size, hidden_size, att_size=100, cell_hid_size=100, num_steps=10, directions=2):
    super(MultiStepAttention, self).__init__()
    self.directions = directions
    self.num_steps = num_steps
    self.attn = Attention(in_size=input_size, hid_size=cell_hid_size, att_size=att_size)
    self.linear_proj = nn.Linear(hidden_size*self.directions, cell_hid_size)
    self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=cell_hid_size)

  def forward(self, x_in, hidden, seq_lengths): #x_in: (batch_size, seq_len, hidden_size*directions)
    h_x, c_x = hidden #h_x: (directions, batch_size, hidden_size)

    # Prepare hidden states for lstm_cell
    h_x = h_x.permute(1,0,2) # (batch_size, directions, hidden_size)
    c_x = c_x.permute(1,0,2)
    h_x = h_x.reshape(h_x.size(0), h_x.size(2)*self.directions) #(batch_size, hidden_size*directions)
    c_x = c_x.reshape(c_x.size(0), c_x.size(2)*self.directions)
    h_x = self.linear_proj(h_x) # (batch_size, cell_hid_size)
    c_x = self.linear_proj(c_x) # (batch_size, cell_hid_size)

    for i in range(self.num_steps):
      c_vector, alpha = self.attn(x_in=x_in, h_prev=h_x, seq_lengths=seq_lengths) # c_vector: (batch_size, hidden_size*2)
      h_x, c_x = self.lstm_cell(input=c_vector, hx=(h_x,c_x)) # h_x: (batch_size, hidden_size*2)

    # output c_vector rather tan h_x to reduce parameters and overfitting
    return c_vector, alpha


class ABLSTM(nn.Module):
  def __init__(self, batch_size, n_hid, n_feat, n_class, lr, drop_per, drop_hid, n_filt, conv_kernel_sizes=[1,3,5,9,15,21], att_size=100, cell_hid_size=100, use_cnn=False):
    super(ABLSTM, self).__init__()
    self.use_cnn = use_cnn
    
    self.in_drop = nn.Dropout2d(drop_per)
    self.drop = nn.Dropout(drop_hid)
    
    if self.use_cnn:
      self.convs = nn.ModuleList([nn.Conv1d(in_channels=n_feat, out_channels=n_filt, kernel_size=i, padding=i//2) for i in conv_kernel_sizes])
      self.cnn_final = nn.Conv1d(in_channels=len(self.convs)*n_filt, out_channels=10*len(self.convs), kernel_size=3, padding= 3//2)
      self.lstm = nn.LSTM(10*len(self.convs), n_hid, bidirectional=True, batch_first=True)
      
    else:
      self.lstm = nn.LSTM(n_feat, n_hid, bidirectional=True, batch_first=True) #input shape: (seq_len, batch_size, feature_size)
    
    self.relu = nn.ReLU()
    self.multi_attn = MultiStepAttention(input_size=n_hid*2, hidden_size=n_hid, att_size=att_size, cell_hid_size=cell_hid_size, num_steps=10, directions=2)
    #self.attn = Attention(n_hid*2, n_hid)
    self.dense = nn.Linear(n_hid*2, n_hid*2)
    self.label = nn.Linear(n_hid*2, n_class)
 
    self.init_weights()
    
    
  def init_weights(self):
    self.dense.bias.data.zero_()
    torch.nn.init.orthogonal_(self.dense.weight.data, gain=math.sqrt(2))
    
    self.label.bias.data.zero_()
    torch.nn.init.orthogonal_(self.label.weight.data, gain=math.sqrt(2))
    
    for m in self.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
          for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data, gain=1)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data, gain=1)
            elif 'bias_ih' in name:
              param.data.zero_()
            elif 'bias_hh' in name:
              param.data.zero_()
        elif type(m) in [nn.Conv1d]:
          for name, param in m.named_parameters():
            if 'weight' in name:
              torch.nn.init.orthogonal_(param.data, gain=math.sqrt(2))           
            if 'bias' in name:
              param.data.zero_()
    
  def forward(self,inp, seq_lengths):
    x = self.in_drop(inp)  # (batch_size, seq_len, feature_size)

    x = x.permute(0, 2, 1)  # (batch_size, feature_size, seq_len)

    if self.use_cnn:
      conv_cat = torch.cat([self.relu(conv(x)) for conv in self.convs], dim=1) # (batch_size, feature_size*len(convs), seq_len)
      x = self.relu(self.cnn_final(conv_cat)) #(batch_size, out_channels=60, seq_len)

    x = x.permute(0, 2, 1) #(batch_size, seq_len, out_channels=60)
    x = self.drop(x)
    
    pack = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
    packed_output, (h, c) = self.lstm(pack) #h = (2, batch_size, hidden_size)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) #(batch_size, seq_len, hidden_size*2)
      
    #attn_output, alpha = self.attn(x_in=output, seq_lengths=seq_lengths) #(batch_size, hidden_size*2) alpha = (batch_size, seq_len)
    attn_output, alpha = self.multi_attn(x_in=output, hidden=(h, c), seq_lengths=seq_lengths) #(batch_size, hidden_size*2) alpha = (batch_size, seq_len)
    output = self.drop(attn_output)
    
    output = self.relu(self.dense(output)) # (batch_size, hidden_size*2)
    output = self.drop(output)
    
    out = self.label(output) #(batch_size, num_classes)
    #out_mem = self.mem(output) #(batch_size, 1)

    return out, (h, c), alpha # (h, c), alpha only used for visualization in notebooks