import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.awd_model import AWD_Embedding
from models.encoders.deeploc_raw import Encoder as BaseEncoder


class Encoder(BaseEncoder):
  """
  Encoder

  Inputs: input, seq_len
    - **input** of shape
  Outputs: output
    - **output** of shape (batch_size, seq_len, hidden_size*2+320)
  """
  def __init__(self, args, awd_layer, architecture):
    super().__init__(args)
    self.awd_layer = awd_layer
    self.architecture = architecture
    self.awd = AWD_Embedding(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True)

    # load pretrained awd
    with open("pretrained_models/awd_lstm/test_v2_statedict.pt", 'rb') as f:
        state_dict = torch.load(f, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    self.awd.load_state_dict(state_dict)

    if awd_layer in ["2ndlast"]:
      self.project = nn.Linear(1280, 300, bias=False)
    elif awd_layer in ["last"]:
      self.project = nn.Linear(320, 300, bias=False)

    if self.architecture in ["before", "both"]:
      self.lstm = nn.LSTM(128+300, args.n_hid, bidirectional=True, batch_first=True)


  def forward(self, inp, seq_lengths):
    #### AWD 
    with torch.no_grad():
      all_hid, _, _ = self.awd(input=inp, seq_lengths=seq_lengths)

    if self.awd_layer == "last":
      awd_hid = all_hid[2]

      awd_hid = awd_hid.permute(1,0,2) # (bs, seq_len, 320)

    elif self.awd_layer == "2ndlast":
      awd_hid = all_hid[1]

      awd_hid = awd_hid.permute(1,0,2) # (bs, seq_len, 1280) 
    
    awd_hid = self.project(awd_hid) # (bs, seq_len, 300)
    ### End AWD 
    
    inp = self.embed(inp) # (batch_size, seq_len, emb_size)

    inp = self.in_drop1d(inp) # feature dropout
    x = self.in_drop2d(inp)  # (batch_size, seq_len, emb_size) - 2d dropout

    x = x.permute(0, 2, 1)  # (batch_size, emb_size, seq_len)
    conv_cat = torch.cat([self.relu(conv(x)) for conv in self.convs], dim=1) # (batch_size, emb_size*len(convs), seq_len)
    x = self.relu(self.cnn_final(conv_cat)) #(batch_size, out_channels=128, seq_len)

    x = x.permute(0, 2, 1) #(batch_size, seq_len, out_channels=128)
    if self.architecture in ["before", "both"]:
      x = torch.cat((x, awd_hid), dim=2)
    x = self.drop(x) #( batch_size, seq_len, lstm_input_size)
    
    pack = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
    packed_output, (h, c) = self.lstm(pack) #h = (2, batch_size, hidden_size)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) #(batch_size, seq_len, hidden_size*2)
  
    if self.architecture in ["after", "both"]:
      output = torch.cat((output, awd_hid), dim=2) # (batch_size, seq_len, hidden_size*2+300)
      
    return output