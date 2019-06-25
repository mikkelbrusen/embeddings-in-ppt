import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import rename_state_dict_keys
from models.utils.elmo_model import Elmo, key_transformation
from models.encoders.deeploc_raw import Encoder as BaseEncoder


class Encoder(BaseEncoder):
  """
  Encoder with elmo concatenated to the LSTM output

  Parameters:
    -- elmo_layer: last or 2ndlast
    -- architecture: before, after or both

  Inputs: input, seq_len
    - **input** of shape
  Outputs: output
    - **output** of shape (batch_size, seq_len, hidden_size*2 + 300) if arch is after/both else 
      (batch_size, seq_len, hidden_size*2)
  """
  def __init__(self, args, direction):
    super().__init__(args)
    self.direction = direction
    
    self.elmo = Elmo(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True)

    with open("pretrained_models/elmo/elmo_parameters_statedict.pt", 'rb') as f:
      state_dict = torch.load(f, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = rename_state_dict_keys(state_dict, key_transformation)
    self.elmo.load_state_dict(state_dict, strict=False)

  def forward(self, inp, seq_lengths):
    with torch.no_grad():
        (all_hid, all_hid_rev) , _, _ = self.elmo(inp, seq_lengths) # all_hid, last_hidden_states, emb
    
    if self.direction == "forward":
      elmo_hid = all_hid[2].permute(1,0,2) # (bs, seq_len, 320) 
    elif self.direction == "backward":
      elmo_hid = all_hid_rev[2].permute(1,0,2) # (bs, seq_len, 320) 
    ### End Elmo 
    
    inp = self.embed(inp) # (batch_size, seq_len, emb_size)

    inp = self.in_drop1d(inp) # feature dropout
    inp = self.in_drop2d(inp)  # (batch_size, seq_len, emb_size) - 2d dropout

    inp = inp.permute(0, 2, 1)  # (batch_size, emb_size, seq_len)
    conv_cat = torch.cat([self.relu(conv(inp)) for conv in self.convs], dim=1) # (batch_size, emb_size*len(convs), seq_len)
    inp = self.relu(self.cnn_final(conv_cat)) #(batch_size, out_channels=128, seq_len)

    inp = inp.permute(0, 2, 1) #(batch_size, seq_len, out_channels=128)      
    inp = self.drop(inp) #( batch_size, seq_len, 128)
    
    pack = nn.utils.rnn.pack_padded_sequence(inp, seq_lengths, batch_first=True)
    packed_output, _ = self.lstm(pack) #h = (2, batch_size, hidden_size)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) #(batch_size, seq_len, hidden_size*2)

    output = torch.cat((output, elmo_hid), dim=2) # (batch_size, seq_len, hidden_size*2+320)
  
    return output