import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import rename_state_dict_keys
from models.utils.elmo_model import Elmo, key_transformation
from models.encoders.deeploc_raw import Encoder as BaseEncoder


class Encoder(nn.Module):
  """
  Encoder that just runs awd

  Inputs: input, seq_len
    - **input** of shape:
  Outputs: output
    - **output** is a tuple: (all_hid, last_hidden_states, emb) - NOT batch first
  """
  def __init__(self, args):
    super().__init__()

    self.elmo = Elmo(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True)

    with open("pretrained_models/elmo/elmo_parameters_statedict.pt", 'rb') as f:
      state_dict = torch.load(f, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = rename_state_dict_keys(state_dict, key_transformation)
    self.elmo.load_state_dict(state_dict, strict=False)

  def forward(self, inp, seq_lengths):

    with torch.no_grad():
        (outputs, outputs_rev), (hidden, hidden_rev), emb = self.elmo(input=inp, seq_lengths=seq_lengths)
    
    return (outputs, outputs_rev), (hidden, hidden_rev), emb