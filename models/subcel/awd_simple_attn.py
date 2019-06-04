import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.subcel.base import Model as BaseModel
from models.subcel.base import Config as BaseConfig
from model_utils.attention import Attention
from model_utils.awd_model import AWD_Embedding

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args)
    self.Model = SimpleAttention

class SimpleAttention(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.awd = AWD_Embedding(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True)

    self.attn = Attention(in_size=1280, att_size=args.att_size)
    self.in_drop = nn.Dropout2d(args.in_dropout1d)
    self.label = nn.Linear(1280, args.num_classes)
    self.mem = nn.Linear(1280, 1)
 
    self.init_weights()

    # load pretrained awd
    with open("pretrained_models/awd_lstm/test_v2_statedict.pt", 'rb') as f:
        state_dict = torch.load(f, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    self.awd.load_state_dict(state_dict)

    # init awd hidden
    self.awd_hidden = [(hid[0].to(args.device),hid[1].to(args.device)) for hid in self.awd.init_hidden(args.batch_size)]
    
  def init_weights(self):  
    self.label.bias.data.zero_()
    torch.nn.init.orthogonal_(self.label.weight.data, gain=math.sqrt(2))

  def forward(self, inp, seq_lengths):
    #### AWD 
    with torch.no_grad():
      all_hid, last_hid, raw_all_hid, dropped_all_hid, emb = self.awd(input=inp, hidden=self.awd_hidden, seq_lengths=seq_lengths)

    model_input = all_hid # (seq_len, bs, emb_size)
    model_input = model_input.permute(1,0,2) # (bs, seq_len, emb_size)

    output = self.in_drop(model_input) #(batch_size, seq_len, emb_size)
    attn_output, alpha = self.attn(x_in=output, seq_lengths=seq_lengths) #(batch_size, hidden_size)
    out = self.label(attn_output) #(batch_size, num_classes)
    out_mem = torch.sigmoid(self.mem(attn_output)) #(batch_size, 1)

    return (out, out_mem), alpha # alpha only used for visualization in notebooks

