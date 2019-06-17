import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.secpred.base import Model as BaseModel
from models.secpred.base import Config as BaseConfig
from model_utils.crf_layer import CRF
from model_utils.elmo_bi import Elmo, key_transformation
from utils import rename_state_dict_keys

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args)
    self.Model = Model

class Model(BaseModel):
  def __init__(self, args):
    super().__init__(args)
    self.elmo = Elmo(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True)

    self.densel1 = nn.Linear(self.args.input_size, self.args.n_l1)
    self.label = nn.Linear(self.args.n_l1, self.args.n_outputs)
    self.relu = nn.ReLU()

    if self.args.crf:
        self.crf = CRF(num_tags=self.args.n_outputs, batch_first=True).double()

    self.init_weights()

    with open("pretrained_models/elmo/elmo_parameters_statedict.pt", 'rb') as f:
      state_dict = torch.load(f, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    # Rename statedict if doing elmo_bi
    state_dict = rename_state_dict_keys(state_dict, key_transformation)
    self.elmo.load_state_dict(state_dict, strict=False)

    
  def init_weights(self):
    self.densel1.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.densel1.weight.data, gain=1.0)

    self.label.bias.data.zero_()
    torch.nn.init.xavier_uniform_(tensor=self.label.weight.data, gain=1.0)
    
  def forward(self, inp, seq_lengths):
    inp = inp.long()
    #### Elmo 
    with torch.no_grad():
      (elmo_hid, elmo_hid_rev), last_hid, raw_all_hid, dropped_all_hid, emb = self.elmo(input=inp, seq_lengths=seq_lengths)
    
    elmo_hid = elmo_hid.permute(1,0,2) # (bs, seq_len, emb_size) 
    elmo_hid_rev = elmo_hid_rev.permute(1,0,2) # (bs, seq_len, emb_size) 
    ### End Elmo 

    emb = torch.cat((elmo_hid, elmo_hid_rev),dim=2)
    output = self.relu(self.densel1(emb))
    out = self.label(output)

    return out