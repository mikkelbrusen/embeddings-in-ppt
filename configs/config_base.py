from abc import abstractmethod
import torch.nn as nn

class Model(nn.Module):
  def __init__(self, args, Encoder, Decoder):
    super().__init__()

    self.args = args
    self.encoder = Encoder(args)
    self.decoder = Decoder(args)

  def forward(self, inp, seq_len):
    output = self.encoder(inp, seq_len)
    output = self.decoder(output, seq_len)

    return output


class Config:
  """
  This is the interface that all config files should implement

  Init: In the init method, one should create network and load in the dataset

  Trainer: In the trainer method, one should train and validate

  Tester: In the test method, one should test
  """

  @abstractmethod
  def __init__(self, args):
    raise NotImplementedError

  @abstractmethod
  def trainer(self):
    """
    Should have the following return statement:
      return best_val_accuracies, best_val_models

    where best_val_models can be one model or a list of models to be used for ensambling
    """
    raise NotImplementedError

  @abstractmethod
  def tester(self):
    raise NotImplementedError