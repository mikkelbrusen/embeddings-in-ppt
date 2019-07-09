import math
import numpy as np
import torch
import sys
import time
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.autograd import Variable
from models.utils.crf_layer import CRF
import dataloaders.secpred as data

from utils.utils import save_model, load_model

from configs.config_base import Config as ConfigBase

class Model(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args

################################
#            Config
################################

class Config(ConfigBase):
  def __init__(self, args, Model):
    self.args = args
    self.Model = Model

    self.train_data_gen, self.validdata, self.testdata, self.num_batch = self._load_data()

  def _load_data(self):
    data_gen = data.gen_data(batch_size=self.args.batch_size, is_cb513=self.args.cb513, is_raw=self.args.raw, profiles_with_raw=self.args.profiles_with_raw, train_path=self.args.trainset, test_path=self.args.testset)
    num_batch = data_gen._num_seq_train // self.args.batch_size
    data_gen_train = data_gen.gen_train(is_raw=self.args.raw)
    validdata = data_gen.get_valid_data()
    testdata = data_gen.get_test_data()

    return data_gen_train, validdata, testdata, num_batch

  def _get_optimizer(self, model, optimizer):
    if optimizer == 'adam':
      return torch.optim.Adam(params=model.parameters(), lr=self.args.learning_rate)
    if optimizer == 'adadelta':
      return torch.optim.Adadelta(params=model.parameters(), lr=self.args.learning_rate)
    raise ValueError("No optimizer found for: {}".format(optimizer))

  def calculate_accuracy_crf(self, preds, targets, mask):
    correct = 0
    for i in range(len(preds)):
        pred = torch.tensor(preds[i]).type(torch.float64).to(self.args.device)
        target = targets[i][mask[i]].type(torch.float64).to(self.args.device)
        correct += torch.sum(pred.eq(target))
    return correct.type(torch.float64) / torch.sum(mask.type(torch.float64))

  def calculate_accuracy(self, preds, targets, mask):
    preds = preds.argmax(2).type(torch.float)
    correct = preds.type(torch.float).eq(targets.type(torch.float)).type(torch.float) * mask.type(torch.float)
    return torch.sum(correct) / torch.sum(mask)

  def run_train(self, model):
    train_err = 0.0
    accuracy = 0.0
    train_batches = 0
    model.train()
    start_time = time.time()
    for b in range(self.num_batch):
      batch = next(self.train_data_gen)
      seq_lengths = batch['length']
      #sort to be in decending order for pad packed to work
      perm_idx = np.argsort(-seq_lengths)
      seq_lengths = seq_lengths[perm_idx]
      inputs = batch['X'][perm_idx]
      targets = batch['t'][perm_idx]
      mask = batch['mask'][perm_idx]
      inp = Variable(torch.from_numpy(inputs).type(torch.float)).to(self.args.device)
      seq_lens = Variable(torch.from_numpy(seq_lengths).type(torch.int32)).to(self.args.device)
      mask_byte = Variable(torch.from_numpy(mask).type(torch.ByteTensor)).to(self.args.device)
      mask_float = Variable(torch.from_numpy(mask).type(torch.float)).to(self.args.device)
      targets = Variable(torch.from_numpy(targets).type(torch.long)).to(self.args.device)

      self.optimizer.zero_grad()
      output = model(inp=inp, seq_len=seq_lens)
      
      if self.args.crf:
          # calculate loss
          output = output.double()
          mask_float = mask_float.double()
          loss = -model.crf(emissions=output, tags=targets, mask=mask_byte)
          loss = loss / torch.sum(mask_float)
          loss.backward()

          # calculate accuaracy
          preds_list = model.crf.decode(emissions=output, mask=mask_byte)
          accuracy += self.calculate_accuracy_crf(preds=preds_list, targets=targets, mask=mask_byte)

          torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.args.clip)
      else:
          # calculate loss
          loss = 0
          loss_preds = output.permute(1,0,2)
          loss_mask = mask_float.permute(1,0)
          loss_targets = targets.permute(1,0)
          for i in range(loss_preds.size(0)):
              loss += torch.sum(F.cross_entropy(loss_preds[i], loss_targets[i], reduction='none') * loss_mask[i])
          loss = loss / (torch.sum(loss_mask)+1e-12)
          loss.backward()

          # calculate accuaracy
          accuracy += self.calculate_accuracy(preds=output, targets=targets, mask=mask_float)

          torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.args.clip)
      
      self.optimizer.step()

      train_err += loss.item()
      train_batches += 1
    train_accuracy = accuracy / train_batches
    train_loss = train_err / train_batches
    return train_loss, train_accuracy

  def run_eval(self, model):
    accuracy = 0.0
    model.eval()
    with torch.no_grad():
      inputs, targets, mask, seq_lengths = self.validdata

      #sort to be in decending order for pad packed to work
      perm_idx = np.argsort(-seq_lengths)
      seq_lengths = seq_lengths[perm_idx]
      inputs = inputs[perm_idx]
      targets = targets[perm_idx]
      mask = mask[perm_idx]
      inp = Variable(torch.from_numpy(inputs).type(torch.float)).to(self.args.device)
      seq_lens = Variable(torch.from_numpy(seq_lengths).type(torch.int32)).to(self.args.device)
      mask_byte = Variable(torch.from_numpy(mask).type(torch.ByteTensor)).to(self.args.device)
      mask_float = Variable(torch.from_numpy(mask).type(torch.float)).to(self.args.device)
      targets = Variable(torch.from_numpy(targets).type(torch.long)).to(self.args.device)

      output = model(inp=inp, seq_len=seq_lens)
      
      if self.args.crf:
          output = output.double()
          mask_float = mask_float.double()
          # calculate loss
          loss = -model.crf(emissions=output, tags=targets, mask=mask_byte)
          loss = loss / torch.sum(mask_float)
          # calculate accuaracy
          preds_list = model.crf.decode(emissions=output, mask=mask_byte)
          accuracy += self.calculate_accuracy_crf(preds=preds_list, targets=targets, mask=mask_byte)
      else:
          # calculate loss
          loss = 0
          loss_preds = output.permute(1,0,2)
          loss_mask = mask_float.permute(1,0)
          loss_targets = targets.permute(1,0)
          for i in range(loss_preds.size(0)):  #try and make into matrix loss
              loss += torch.sum(F.cross_entropy(loss_preds[i], loss_targets[i], reduction='none') * loss_mask[i])
          loss = loss / (torch.sum(loss_mask)+1e-12)
          # calculate accuaracy
          accuracy += self.calculate_accuracy(preds=output, targets=targets, mask=mask_float)

      return loss, accuracy

  def run_test(self, model):
    accuracy = 0.0
    model.eval()
    with torch.no_grad():
      inputs, targets, mask, seq_lengths = self.testdata

      #sort to be in decending order for pad packed to work
      perm_idx = np.argsort(-seq_lengths)
      seq_lengths = seq_lengths[perm_idx]
      inputs = inputs[perm_idx]
      targets = targets[perm_idx]
      mask = mask[perm_idx]
      inp = Variable(torch.from_numpy(inputs).type(torch.float)).to(self.args.device)
      seq_lens = Variable(torch.from_numpy(seq_lengths).type(torch.int32)).to(self.args.device)
      mask_byte = Variable(torch.from_numpy(mask).type(torch.ByteTensor)).to(self.args.device)
      mask_float = Variable(torch.from_numpy(mask).type(torch.float)).to(self.args.device)
      targets = Variable(torch.from_numpy(targets).type(torch.long)).to(self.args.device)

      output = model(inp=inp, seq_len=seq_lens)
      
      if self.args.crf:
          output = output.double()
          mask_float = mask_float.double()
          # calculate loss
          loss = -model.crf(emissions=output, tags=targets, mask=mask_byte)
          loss = loss / torch.sum(mask_float)
          # calculate accuaracy
          preds_list = model.crf.decode(emissions=output, mask=mask_byte)
          accuracy += self.calculate_accuracy_crf(preds=preds_list, targets=targets, mask=mask_byte)
      else:
          # calculate loss
          loss = 0
          loss_preds = output.permute(1,0,2)
          loss_mask = mask_float.permute(1,0)
          loss_targets = targets.permute(1,0)
          for i in range(loss_preds.size(0)):  #try and make into matrix loss
              loss += torch.sum(F.cross_entropy(loss_preds[i], loss_targets[i], reduction='none') * loss_mask[i])
          loss = loss / (torch.sum(loss_mask)+1e-12)
          # calculate accuaracy
          accuracy += self.calculate_accuracy(preds=output, targets=targets, mask=mask_float)

      return loss, accuracy

  def trainer(self):
    model = self.Model(self.args).to(self.args.device)
    self.optimizer = self._get_optimizer(model=model, optimizer=self.args.optimizer)
    print("Model: ", model)

    best_val_acc = 0.0
    best_val_loss = 100000000
    idx = 0
    for epoch in range(self.args.epochs):
        start_time = time.time()
        train_loss, train_accuracy = self.run_train(model=model)
        val_loss, val_accuracy = self.run_eval(model=model)
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            best_val_acc = val_accuracy
            idx = epoch
            save_model(model, self.args)

        print('-' * 22, ' epoch: {:3d} / {:3d} - time: {:5.2f}s '.format(epoch, self.args.epochs, time.time() - start_time), '-' * 22 )
        #Train
        print('| Train | loss {:.4f} | acc {:.2f}%'
        ' |'.format(train_loss, train_accuracy*100))
        print('| Valid | loss {:.4f} | acc {:.2f}%' 
        ' |'.format(val_loss, val_accuracy*100))
        sys.stdout.flush()

    print('BEST RESULTS')
    print('| Valid | epoch {:3d} | acc {:.2f}%'
    ' |'.format(idx, best_val_acc*100))
    sys.stdout.flush()

  def tester(self):
    model = self.Model(self.args).to(self.args.device)
    load_model(model, self.args)

    test_loss, test_accuracy = self.run_test(model=model)

    print('| Test | loss {:.4f} | acc {:.2f}%' 
    ' |'.format(test_loss, test_accuracy*100))
    print('-' * 79)