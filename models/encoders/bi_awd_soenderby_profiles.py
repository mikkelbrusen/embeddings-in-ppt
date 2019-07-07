import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import init_weights
from utils.utils import rename_state_dict_keys, init_weights
from models.utils.bi_awd_model import BiAWDEmbedding, key_transformation

class Encoder(nn.Module):
  def __init__(self, args, bi_awd_layer, architecture):
    super().__init__()
    self.args = args
    self.densel1 = nn.Linear(self.args.n_features, self.args.n_l1)
    self.densel2 = nn.Linear(self.args.n_l1, self.args.n_l1)
    self.bi_rnn = nn.LSTM(input_size=self.args.n_l1+self.args.n_features, hidden_size=self.args.n_hid, num_layers=3, bidirectional=True, batch_first=True)
    self.drop = nn.Dropout(p=0.5)
    self.relu = nn.ReLU()

    if bi_awd_layer in ["second"]:
      self.project = nn.Linear(2560, 300, bias=False)
    elif bi_awd_layer in ["last"]:
      self.project = nn.Linear(320*2, 300, bias=False)

    if self.architecture in ["before", "both"]:
      self.lstm = nn.LSTM(128+300, args.n_hid, bidirectional=True, batch_first=True)

    init_weights(self)
    self.init_weights()

    self.bi_awd = BiAWDEmbedding(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True)
    self.bi_awd.load_pretrained()
   
  def init_weights(self):
    self.densel1.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.densel1.weight.data, gain=1.0)

    self.densel2.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.densel2.weight.data, gain=1.0)
    
  def forward(self, inp, seq_lengths):
    #Something like this. Look into it when needed
    profiles, raw = inp
    with torch.no_grad():
      (all_hid, all_hid_rev) , _, _ = self.bi_awd(inp, seq_lengths) # all_hid, last_hidden_states, emb
    
    if self.bi_awd_layer == "last":
      bi_awd_hid = all_hid[2]
      bi_awd_hid_rev = all_hid_rev[2]

      bi_awd_hid = bi_awd_hid.permute(1,0,2) # (bs, seq_len, 320) 
      bi_awd_hid_rev = bi_awd_hid_rev.permute(1,0,2) # (bs, seq_len, 320) 

    elif self.bi_awd_layer == "second":
      bi_awd_hid = all_hid[1]
      bi_awd_hid_rev = all_hid_rev[1]

      bi_awd_hid = bi_awd_hid.permute(1,0,2) # (bs, seq_len, 1280) 
      bi_awd_hid_rev = bi_awd_hid_rev.permute(1,0,2) # (bs, seq_len, 1280) 
    
    bi_awd_hid = torch.cat((bi_awd_hid, bi_awd_hid_rev), dim=2) # (bs, seq_len, something big) 
    bi_awd_hid = self.project(bi_awd_hid) # (bs, seq_len, 300) 
    del bi_awd_hid_rev
    ### End BiAWDEmbedding

    x = self.relu(self.densel2(self.relu(self.densel1(inp))))
    x = torch.cat((inp,x), dim=2)
    if self.architecture in ["before", "both"]:
      inp = torch.cat((x, bi_awd_hid), dim=2)
    pack = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
    packed_output, _ = self.bi_rnn(pack)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

    if self.architecture in ["after", "both"]:
      output = torch.cat((output, bi_awd_hid), dim=2) # (batch_size, seq_len, hidden_size*2+300)

    return output