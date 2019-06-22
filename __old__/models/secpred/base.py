import math
import numpy as np
import torch
import sys
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
  def __init__(self, args):
    super().__init__()
    
    
  def init_weights(self):
    pass
    
  def forward(self, inp, seq_lengths):
    pass


################################
#            Config
################################

class Config:
  def __init__(self, args):
    self.args = args
    self.Model = Model


  def _load_data(self):
    pass

  def run_train(self, model, X, y, mask, mem, unk):
    pass

  def run_eval(self, model, X, y, mask, mem, unk):
    pass

  def run_test(self, models, X, y, mask, mem, unk):
    pass

  def trainer(self):
    model = self.Model(self.args).to(self.args.device)
    print("Model: ", model)

    best_val_accs = []
    best_val_models = []

    return best_val_accs, best_val_models


  def tester(self, best_val_models):
    self.run_test(best_val_models, None, None, None, None, None)
    pass