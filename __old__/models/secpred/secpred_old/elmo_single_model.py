import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import rename_state_dict_keys
from models.secpred.base import Model as BaseModel
from models.secpred.base import Config as BaseConfig
from model_utils.elmo_bi import Elmo, key_transformation

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args)
    self.Model = Model

class Model(BaseModel):
  def __init__(self, args):
    super().__init__(args)
    self.embed = nn.Embedding(num_embeddings=21, embedding_dim=self.args.input_size, padding_idx=20)
    self.elmo = Elmo(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True)

    self.in_drop1d = nn.Dropout(0.2)
    self.in_drop2d = nn.Dropout2d(0.2)
    self.drop = nn.Dropout(0.5)

    self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.args.input_size, out_channels=self.args.input_size, kernel_size=i, padding=i//2) for i in [1,3,5,9,15,21]])
    self.cnn_final = nn.Conv1d(in_channels=len(self.convs)*self.args.input_size, out_channels=128, kernel_size=3, padding= 3//2)
    self.relu = nn.ReLU()

    # modify base layers to accomodate the concatenated states
    self.lstm = nn.LSTM(128, self.args.n_rnn_hid, bidirectional=True, batch_first=True)

    self.dense = nn.Linear(self.args.n_rnn_hid*2+320*2, self.args.n_rnn_hid*2+320*2)
    self.label = nn.Linear(self.args.n_rnn_hid*2+320*2, self.args.n_outputs)
    self.init_weights()

    # load pretrained elmo
    with open("pretrained_models/elmo/elmo_parameters_statedict.pt", 'rb') as f:
      state_dict = torch.load(f, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = rename_state_dict_keys(state_dict, key_transformation)
    self.elmo.load_state_dict(state_dict, strict=False)

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

  def forward(self, inp, seq_lengths):
    inp = inp.long()
    #### Elmo 
    with torch.no_grad():
      (elmo_hid, elmo_hid_rev), last_hid, raw_all_hid, dropped_all_hid, emb = self.elmo(input=inp, seq_lengths=seq_lengths)
    
    elmo_hid = elmo_hid.permute(1,0,2) # (bs, seq_len, emb_size) 
    elmo_hid_rev = elmo_hid_rev.permute(1,0,2) # (bs, seq_len, emb_size) 
    ### End Elmo 
    
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

    ### Concat 320 hidden layer to BiLSTM
    output = torch.cat((output, elmo_hid, elmo_hid_rev),dim=2) # (batch_size, seq_len, hidden_size*2+320*2)
    ### End Concat
    
    output = self.relu(self.dense(output)) # (batch_size, hidden_size*2)
    output = self.drop(output)
    
    out = self.label(output) #(batch_size, num_classes)

    return out # alpha only used for visualization in notebooks