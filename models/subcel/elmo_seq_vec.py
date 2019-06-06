import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import reverse_padded_sequence, rename_state_dict_keys
from model_utils.attention import Attention, MultiStepAttention
from models.subcel.base import Model as BaseModel
from models.subcel.base import Config as BaseConfig
from model_utils.elmo import Elmo

# This is used for elmo_bi where we use bidirectional LSTMs rather than unidirectional LSTMs on reversed input

# def key_transformation(old_key: str):
#     if "rnns_rev" in old_key:
#         old_key = "rnns" + old_key.split("rnns_rev")[1] + "_reverse"

#     if "_raw_reverse" in old_key:
#         old_key = old_key.split("_raw_reverse")[0] + "_reverse_raw"

#     return old_key


class Config(BaseConfig):
    def __init__(self, args):
        super().__init__(args)
        self.Model = Model

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.elmo = Elmo(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True)

        self.linear = nn.Linear(1280, 32)
        self.drop = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(32)
        self.label = nn.Linear(32, args.num_classes)
        self.mem = nn.Linear(32, 1)
        
        self.init_weights()

        # load pretrained elmo
        with open("pretrained_models/elmo/elmo_parameters_statedict.pt", 'rb') as f:
            state_dict = torch.load(f, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        self.elmo.load_state_dict(state_dict)

    def init_weights(self):
        self.linear.bias.data.zero_()
        torch.nn.init.orthogonal_(self.linear.weight.data, gain=math.sqrt(2))
        
    def forward(self, inp, seq_lengths):
        #### Elmo 
        inp_rev = reverse_padded_sequence(inp, seq_lengths, batch_first=True)
        with torch.no_grad():
            all_hid, last_hid, raw_all_hid, dropped_all_hid, emb = self.elmo(input=inp, input_rev=inp_rev, seq_lengths=seq_lengths)
        
        (elmo_hid, elmo_hid_rev) = dropped_all_hid # [(seq_len, bs, 1280),(seq_len, bs, 1280),(seq_len, bs, emb_size)] , ...
        ### End Elmo 
        
        model_input = (elmo_hid[0] + elmo_hid[1] + elmo_hid_rev[0] + elmo_hid_rev[1]) #(seq_len, bs, 1280)
        model_input = model_input.mean(0) # (bs, 1280)

        output = self.bn(self.relu(self.drop(self.linear(model_input))))

        out = self.label(output) #(batch_size, num_classes)
        out_mem = torch.sigmoid(self.mem(output)) #(batch_size, 1)

        return (out, out_mem), None # alpha only used for visualization in notebooks