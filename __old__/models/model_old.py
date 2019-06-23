import math
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
sys.path.insert(0,'..')
from models.utils.attention import Attention, MultiStepAttention
from utils.utils import do_layer_norm


class ABLSTM(nn.Module):
  def __init__(self, batch_size, n_hid, n_feat, n_class, drop_per, drop_hid, n_filt, conv_kernel_sizes=[1,3,5,9,15,21], att_size=256, 
  cell_hid_size=512, num_steps=10, directions=2, is_multi_step=True):
    super(ABLSTM, self).__init__()
    self.is_multi_step = is_multi_step

    self.in_drop = nn.Dropout2d(drop_per)
    self.drop = nn.Dropout(drop_hid)

    self.embed = nn.Embedding(num_embeddings=21, embedding_dim=n_feat, padding_idx=20)

    self.convs = nn.ModuleList([nn.Conv1d(in_channels=n_feat, out_channels=n_filt, kernel_size=i, padding=i//2) for i in conv_kernel_sizes])
    self.cnn_final = nn.Conv1d(in_channels=len(self.convs)*n_filt, out_channels=128, kernel_size=3, padding= 3//2)
    self.lstm = nn.LSTM(128, n_hid, bidirectional=True, batch_first=True)
    self.lstm2 = nn.LSTM(n_hid*2+320, n_hid, bidirectional=True, batch_first=True)
    
    self.relu = nn.ReLU()
    if (is_multi_step):
      self.attn = MultiStepAttention(input_size=n_hid*2, hidden_size=n_hid, att_size=att_size, cell_hid_size=cell_hid_size, num_steps=num_steps, directions=directions)
    else:
      self.attn = Attention(in_size=n_hid*2, att_size=att_size)

    self.dense = nn.Linear(n_hid*2, n_hid*2)
    self.label = nn.Linear(n_hid*2, n_class)
    self.mem = nn.Linear(n_hid*2, 1)
 
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
    
  def forward(self, inp, seq_lengths, mask, awd_hid):
    #x = inp
    inp = self.embed(inp) # (batch_size, seq_len, emb_size)

    # Concat 320 hidden layer to embedding
    #normed_awd_hid = do_layer_norm(awd_hid, mask) # (batch_size, seq_len, 320)
    #inp = torch.cat((inp, normed_awd_hid), dim=2) # (batch_size, seq_len, 320+emb_size)

    inp = self.drop(inp) # feature dropout
    x = self.in_drop(inp)  # (batch_size, seq_len, emb_size) - 2d dropout

    x = x.permute(0, 2, 1)  # (batch_size, emb_size, seq_len)
    conv_cat = torch.cat([self.relu(conv(x)) for conv in self.convs], dim=1) # (batch_size, emb_size*len(convs), seq_len)
    x = self.relu(self.cnn_final(conv_cat)) #(batch_size, out_channels=128, seq_len)

    x = x.permute(0, 2, 1) #(batch_size, seq_len, out_channels=128)
    x = self.drop(x) #( batch_size, seq_len, lstm_input_size)
    
    pack = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
    packed_output, (h, c) = self.lstm(pack) #h = (2, batch_size, hidden_size)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) #(batch_size, seq_len, hidden_size*2)

    # Concat 320 hidden layer to BiLSTM
    output = torch.cat((output, awd_hid),dim=2) # (batch_size, seq_len, hidden_size*2+320)

    pack = nn.utils.rnn.pack_padded_sequence(output, seq_lengths, batch_first=True)
    packed_output, (h, c) = self.lstm2(pack) #h = (2, batch_size, hidden_size)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) #(batch_size, seq_len, hidden_size*2)
  
    attn_output, alpha = self.attn(x_in=output, seq_lengths=seq_lengths) #(batch_size, hidden_size*2) alpha = (batch_size, seq_len)
    output = self.drop(attn_output)
    
    output = self.relu(self.dense(output)) # (batch_size, hidden_size*2)
    output = self.drop(output)
    
    out = self.label(output) #(batch_size, num_classes)
    out_mem = torch.sigmoid(self.mem(output)) #(batch_size, 1)

    return (out, out_mem), alpha # alpha only used for visualization in notebooks

class SimpleAttention(nn.Module):
  def __init__(self, batch_size, n_hid, n_class, drop_per, att_size=256):
    super(SimpleAttention, self).__init__()
    self.attn = Attention(in_size=n_hid, att_size=att_size)
    self.in_drop = nn.Dropout2d(drop_per)
    self.label = nn.Linear(n_hid, n_class)
    self.mem = nn.Linear(n_hid, 1)
 
    self.init_weights()
    
  def init_weights(self):  
    self.label.bias.data.zero_()
    torch.nn.init.orthogonal_(self.label.weight.data, gain=math.sqrt(2))

  def forward(self, inp, seq_lengths):
    output = self.in_drop(inp) #(batch_size, seq_len, emb_size)
    attn_output, alpha = self.attn(x_in=output, seq_lengths=seq_lengths) #(batch_size, hidden_size)
    out = self.label(attn_output) #(batch_size, num_classes)
    out_mem = torch.sigmoid(self.mem(attn_output)) #(batch_size, 1)

    return (out, out_mem), alpha # alpha only used for visualization in notebooks

class StraightToLinear(nn.Module):
  def __init__(self, batch_size, n_hid, n_class, drop_per, att_size=256):
    super(StraightToLinear, self).__init__()

    #self.drop = nn.Dropout(drop_per)
    self.relu = nn.ReLU()
    self.dense1 = nn.Linear(n_hid, n_hid)
    #self.dense2 = nn.Linear(n_hid, n_hid)
    #self.dense3 = nn.Linear(n_hid, n_hid)
    self.label = nn.Linear(n_hid, n_class)
    self.mem = nn.Linear(n_hid, 1)
 
    #self.init_weights()
    
    
  def init_weights(self):
    self.dense1.bias.data.zero_()
    torch.nn.init.orthogonal_(self.dense1.weight.data, gain=math.sqrt(2))
    #self.dense2.bias.data.zero_()
    #torch.nn.init.orthogonal_(self.dense2.weight.data, gain=math.sqrt(2))
    #self.dense3.bias.data.zero_()
    #torch.nn.init.orthogonal_(self.dense3.weight.data, gain=math.sqrt(2)) 
    #self.label.bias.data.zero_()
    #torch.nn.init.orthogonal_(self.label.weight.data, gain=math.sqrt(2))

  def forward(self, inp, seq_lengths):

    output = self.relu(self.dense1(inp))
    #output = self.drop(output)
    #output = self.relu(self.dense2(output))
    #output = self.relu(self.dense3(output))
    out = self.label(output) #(batch_size, num_classes)
    out_mem = torch.sigmoid(self.mem(output)) #(batch_size, 1)

    return (out, out_mem), None # alpha only used for visualization in notebooks

class SeqVec(nn.Module):
  def __init__(self, batch_size, inp_size, n_hid, n_class, drop_per):
    super(SeqVec, self).__init__()
    self.linear = nn.Linear(inp_size, n_hid)
    self.drop = nn.Dropout(drop_per)
    self.relu = nn.ReLU()
    self.bn = nn.BatchNorm1d(32)
    self.label = nn.Linear(n_hid, n_class)
    self.mem = nn.Linear(n_hid, 1)
    
    self.init_weights()
    
  def init_weights(self):
    self.linear.bias.data.zero_()
    torch.nn.init.orthogonal_(self.linear.weight.data, gain=math.sqrt(2))

  def forward(self, inp, seq_lengths):

    output = self.bn(self.relu(self.drop(self.linear(inp))))

    out = self.label(output) #(batch_size, num_classes)
    out_mem = torch.sigmoid(self.mem(output)) #(batch_size, 1)

    return (out, out_mem), None # alpha only used for visualization in notebooks