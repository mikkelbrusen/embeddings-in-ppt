import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import length_to_mask

class Attention(nn.Module):
  def __init__(self, in_size, att_size):
    super(Attention, self).__init__()
    self.linear_in = nn.Linear(in_size, att_size)
    self.linear_att = nn.Linear(att_size, 1, bias=False)
    
  def forward(self, x_in, h_prev, seq_lengths):  # x_in.shape: [bs, seq_len, in_size]
    att_vector = self.linear_in(x_in) # [bs, seq_len, att_size]
    hid_vector = self.linear_hid(h_prev) # bs,att_size

    att_hid_align = torch.tanh(att_vector + hid_vector.unsqueeze(dim=1))
   
    att_score = self.linear_att(att_vector).squeeze(2) # [bs, seq_len]
    
    mask = length_to_mask(seq_lengths)

    alpha = F.softmax(att_score + mask, dim=1) # [bs, seq_len]
    att = alpha.view(x_in.size(0), x_in.size(1), 1) # [bs, seq_len, 1]  att.unsqeeze(2)

    return torch.sum(x_in * att, dim=1), alpha # [bs, in_size]

class MultiStepAttention(nn.Module):
  def __init__(self, input_size, hidden_size, num_steps):
    super(MultiStepAttention, self).__init__()
    self.num_steps = num_steps
    self.attn = Attention(input_size, input_size//2)
    self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=2*hidden_size)

  def forward(self, x_in, hidden, seq_lengths): #x_in: (batch_size, seq_len, hidden_size*2)
    h_x,c_x = hidden #h_x: (2, batch_size, hidden_size)
    h_x = h_x.view(h_x.size(1),h_x.size(2)*2) #(1, batch_size, hidden_size*2)
    c_x = c_x.view(c_x.size(1),c_x.size(2)*2)
#    print("h_x.shape", h_x.shape)
#    print("x_in", x_in.shape)
    for i in range(self.num_steps):
#      print("c_t.shape", c_t.shape)
      c_t, alpha = self.attn(x_in, h_x, seq_lengths+1)
#      print("c_t.shape", c_t.shape)
      h_x, c_x = self.lstm_cell(c_t, (h_x,c_x))

    return h_x, alpha


class ABLSTM(nn.Module):
  def __init__(self, batch_size, n_hid, n_feat, n_class, lr, drop_per, drop_hid, n_filt, use_cnn=False):
    super(ABLSTM, self).__init__()
    self.use_cnn = use_cnn
    
    self.in_drop = nn.Dropout2d(drop_per)
    self.drop = nn.Dropout(drop_hid)
    
    if self.use_cnn:
      self.cnn_a = nn.Conv1d(in_channels=n_feat, out_channels=n_filt, kernel_size=1, padding= 1//2)
      self.cnn_b = nn.Conv1d(in_channels=n_feat, out_channels=n_filt, kernel_size=3, padding= 3//2)
      self.cnn_c = nn.Conv1d(in_channels=n_feat, out_channels=n_filt, kernel_size=5, padding= 5//2)
      #self.cnn_d = nn.Conv1d(in_channels=n_feat, out_channels=n_filt, kernel_size=9, padding= 9//2)
      #self.cnn_e = nn.Conv1d(in_channels=n_feat, out_channels=n_filt, kernel_size=15, padding= 15//2)
      #self.cnn_f = nn.Conv1d(in_channels=n_feat, out_channels=n_filt, kernel_size=21, padding= 21//2)
      self.cnn_final = nn.Conv1d(in_channels=3*n_filt, out_channels=30, kernel_size=3, padding= 3//2)
      
      self.lstm = nn.LSTM(30, n_hid, bidirectional=True, batch_first=True)
      
    else:
      self.lstm = nn.LSTM(n_feat, n_hid, bidirectional=True, batch_first=True) #input shape: (seq_len, batch_size, feature_size)
    
    self.relu = nn.ReLU()
    self.multi_attn = MultiStepAttention(n_hid*2, n_hid, 10)
    #self.attn = Attention(n_hid*2, n_hid) #second param can be small, like n_hid*1
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
      #print("x: ", x.shape)
      conv_a = self.relu(self.cnn_a(x)) # (seq_len, feature_size, batch_size)
      conv_b = self.relu(self.cnn_b(x))
      conv_c = self.relu(self.cnn_c(x))
      #conv_d = self.relu(self.cnn_d(x))
      #conv_e = self.relu(self.cnn_e(x))
      #conv_f = self.relu(self.cnn_f(x))

      #print("conv_a.shape: ", conv_a.shape)
      #conv_cat = torch.cat((conv_a, conv_b, conv_c, conv_d, conv_e, conv_f), dim=1) #conv_cat.shape = (batch_size, feature_size, seq_len)
      conv_cat = torch.cat((conv_a, conv_b, conv_c), dim=1) #conv_cat.shape = (batch_size, feature_size, seq_len)
      #print("conv_cat.shape: ", conv_cat.shape)

      x = self.relu(self.cnn_final(conv_cat))
      #print("conv_final: ", conv_final.shape)

    x = x.permute(0,2,1) #(batch_size, seq_len, feature_size)
    x = self.drop(x)
    
    pack = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
    packed_output, (h, c) = self.lstm(pack) #h = (2, batch_size, hidden_size)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) #(batch_size, seq_len, hidden_size*2)

    #print("h.shape", h.shape)
    #print("h.view", h.view(h.size(1),-1).unsqueeze(1).shape)
      
    #attn_output, alpha = self.attn(x_in=output, seq_lengths=seq_lengths) #(batch_size, hidden_size*2) alpha = (batch_size, seq_len)
    attn_output, alpha = self.multi_attn(x_in=output, hidden=(h, c), seq_lengths=seq_lengths) #(batch_size, hidden_size*2) alpha = (batch_size, seq_len)
    output = self.drop(attn_output)
    
    output = self.relu(self.dense(output)) # (batch_size, hidden_size*2)
    output = self.drop(output)
    
    out = self.label(output) #(batch_size, num_classes)
    #out_mem = self.mem(output) #(batch_size, 1)

    return out, (h, c), alpha