import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.subcel.base_raw import Model as BaseModel
from models.subcel.base_raw import Config as BaseConfig
from models.utils.awd_model import AWD_Embedding

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args)
    self.Model = SeqVec

class SeqVec(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.awd = AWD_Embedding(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True)

    self.linear = nn.Linear(1280, 32)
    self.drop = nn.Dropout(0.2)
    self.relu = nn.ReLU()
    self.bn = nn.BatchNorm1d(32)
    self.label = nn.Linear(32, args.num_classes)
    self.mem = nn.Linear(32, 1)
    
    self.init_weights()

    # load pretrained awd
    with open("pretrained_models/awd_lstm/test_v2_statedict.pt", 'rb') as f:
        state_dict = torch.load(f, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    self.awd.load_state_dict(state_dict)

    # init awd hidden
    self.awd_hidden = [(hid[0].to(args.device),hid[1].to(args.device)) for hid in self.awd.init_hidden(args.batch_size)]
    
  def init_weights(self):
    self.linear.bias.data.zero_()
    torch.nn.init.orthogonal_(self.linear.weight.data, gain=math.sqrt(2))

  def forward(self, inp, seq_lengths):
    #### AWD 
    with torch.no_grad():
      all_hid, last_hid, raw_all_hid, dropped_all_hid, emb = self.awd(input=inp, hidden=self.awd_hidden, seq_lengths=seq_lengths)

    model_input = (dropped_all_hid[0] + dropped_all_hid[1]) #(seq_len, bs, 1280)
    model_input = model_input.mean(0) # (bs, 1280)

    output = self.bn(self.relu(self.drop(self.linear(model_input))))

    out = self.label(output) #(batch_size, num_classes)
    out_mem = torch.sigmoid(self.mem(output)) #(batch_size, 1)

    return (out, out_mem), None # alpha only used for visualization in notebooks