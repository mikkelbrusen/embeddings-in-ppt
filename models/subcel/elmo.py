import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import reverse_padded_sequence
from model_utils.attention import Attention, MultiStepAttention
from models.subcel.base import Model as BaseModel
from models.subcel.base import Config as BaseConfig
from model_utils.elmo import Elmo

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args)
    self.Model = Model

class Model(BaseModel):
  def __init__(self, args):
    super().__init__(args)
    self.elmo = Elmo(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True)

    # modify base layers to accomodate the concatenated states
    self.lstm = nn.LSTM(128, args.n_hid, bidirectional=True, batch_first=True)
    self.attn = Attention(in_size=args.n_hid*2+320*2, att_size=args.att_size)

    self.dense = nn.Linear(args.n_hid*2+320*2, args.n_hid*2+320*2)
    self.label = nn.Linear(args.n_hid*2+320*2, args.num_classes)
    self.mem = nn.Linear(args.n_hid*2+320*2, 1)

    self.init_weights()

    # load pretrained elmo
    with open("pretrained_models/elmo/elmo_parameters_statedict.pt", 'rb') as f:
        state_dict = torch.load(f, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    self.elmo.load_state_dict(state_dict)

  def forward(self, inp, seq_lengths):
    #### Elmo 
    inp_rev = reverse_padded_sequence(inp, seq_lengths, batch_first=True)
    with torch.no_grad():
      all_hid, last_hid, raw_all_hid, dropped_all_hid, emb = self.elmo(input=inp, input_rev=inp_rev, seq_lengths=seq_lengths)
    
    (elmo_hid, elmo_hid_rev) = all_hid # ((seq_len, bs, emb_size),(seq_len, bs, emb_size))
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
    output = torch.cat((output, elmo_hid, elmo_hid_rev),dim=2) # (batch_size, seq_len, hidden_size*2+320)
    ### End Concat
  
    attn_output, alpha = self.attn(x_in=output, seq_lengths=seq_lengths) #(batch_size, hidden_size*2) alpha = (batch_size, seq_len)
    output = self.drop(attn_output)
    
    output = self.relu(self.dense(output)) # (batch_size, hidden_size*2)
    output = self.drop(output)
    
    out = self.label(output) #(batch_size, num_classes)
    out_mem = torch.sigmoid(self.mem(output)) #(batch_size, 1)

    return (out, out_mem), alpha # alpha only used for visualization in notebooks