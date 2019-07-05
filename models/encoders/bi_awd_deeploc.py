import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import rename_state_dict_keys, init_weights
from models.utils.bi_awd_model import BiAWDEmbedding, key_transformation
from models.encoders.deeploc_raw import Encoder as BaseEncoder


class Encoder(BaseEncoder):
  """
  Encoder with elmo concatenated to the LSTM output

  Parameters:
    -- bi_awd_layer: last or 2ndlast
    -- architecture: before, after or both

  Inputs: input, seq_len
    - **input** of shape
  Outputs: output
    - **output** of shape (batch_size, seq_len, hidden_size*2 + 300) if arch is after/both else 
      (batch_size, seq_len, hidden_size*2)
  """
  def __init__(self, args, bi_awd_layer, architecture):
    super().__init__(args)
    self.architecture = architecture
    self.bi_awd_layer = bi_awd_layer

    if bi_awd_layer in ["2ndlast"]:
      self.project = nn.Linear(2560, 300, bias=False)
    elif bi_awd_layer in ["last"]:
      self.project = nn.Linear(320*2, 300, bias=False)

    if self.architecture in ["before", "both"]:
      self.lstm = nn.LSTM(128+300, args.n_hid, bidirectional=True, batch_first=True)

    init_weights(self)

    self.bi_awd = BiAWDEmbedding(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True)
    self.bi_awd.load_pretrained()

  def forward(self, inp, seq_lengths):
    with torch.no_grad():
        (all_hid, all_hid_rev) , _, _ = self.bi_awd(inp, seq_lengths) # all_hid, last_hidden_states, emb
    
    if self.bi_awd_layer == "last":
      elmo_hid = all_hid[2]
      elmo_hid_rev = all_hid_rev[2]

      elmo_hid = elmo_hid.permute(1,0,2) # (bs, seq_len, 320) 
      elmo_hid_rev = elmo_hid_rev.permute(1,0,2) # (bs, seq_len, 320) 

    elif self.bi_awd_layer == "2ndlast":
      elmo_hid = all_hid[1]
      elmo_hid_rev = all_hid_rev[1]

      elmo_hid = elmo_hid.permute(1,0,2) # (bs, seq_len, 1280) 
      elmo_hid_rev = elmo_hid_rev.permute(1,0,2) # (bs, seq_len, 1280) 
    
    elmo_hid = torch.cat((elmo_hid, elmo_hid_rev), dim=2) # (bs, seq_len, something big) 
    elmo_hid = self.project(elmo_hid) # (bs, seq_len, 300) 
    del elmo_hid_rev
    ### End BiAWDEmbedding 
    
    inp = self.embed(inp) # (batch_size, seq_len, emb_size)

    inp = self.in_drop1d(inp) # feature dropout
    inp = self.in_drop2d(inp)  # (batch_size, seq_len, emb_size) - 2d dropout

    inp = inp.permute(0, 2, 1)  # (batch_size, emb_size, seq_len)
    conv_cat = torch.cat([self.relu(conv(inp)) for conv in self.convs], dim=1) # (batch_size, emb_size*len(convs), seq_len)
    inp = self.relu(self.cnn_final(conv_cat)) #(batch_size, out_channels=128, seq_len)

    inp = inp.permute(0, 2, 1) #(batch_size, seq_len, out_channels=128)
    if self.architecture in ["before", "both"]:
      inp = torch.cat((inp, elmo_hid), dim=2)
      
    inp = self.drop(inp) #( batch_size, seq_len, lstm_input_size)
    
    pack = nn.utils.rnn.pack_padded_sequence(inp, seq_lengths, batch_first=True)
    packed_output, _ = self.lstm(pack) #h = (2, batch_size, hidden_size)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) #(batch_size, seq_len, hidden_size*2)

    if self.architecture in ["after", "both"]:
      output = torch.cat((output, elmo_hid), dim=2) # (batch_size, seq_len, hidden_size*2+300)
  
    return output