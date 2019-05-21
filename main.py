import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import os
import time
import argparse 
import sys
import pickle
from importlib import import_module

from utils import iterate_minibatches, ResultsContainer, tensor_to_onehot
from confusionmatrix import ConfusionMatrix
from metrics_mc import gorodkin, IC
from models.model import ABLSTM, StraightToLinear, SeqVec
from models.awd_model import AWD_Embedding
from datautils.dataloader import tokenize_sequence

parser = argparse.ArgumentParser()
# Base parser for all
base_parser = argparse.ArgumentParser(add_help=False)
base_parser.add_argument('--batch_size',  help="Minibatch size", type=int, default=128)
base_parser.add_argument('--epochs',  help="Number of training epochs", type=int, default=250)
base_parser.add_argument('--learning_rate',  help="Learning rate", type=int, default=0.0005)
base_parser.add_argument('--seed',  help="Seed for random number init.", type=int, default=123456)
base_parser.add_argument('--clip', help="Gradient clipping", type=int, default=2)
current_time = time.strftime('%b_%d-%H_%M') # 'Oct_18-09:03'
base_parser.add_argument('--save', help="Path to best saved model", default="save/best_model_" + current_time + ".pt")
base_parser.add_argument('--save_results', help="Path to result object containing all kind of results", default="save/best_results_" + current_time)

### SUBPARSERS ### 
subparsers = parser.add_subparsers()

# Subcellular
parser_subcel = subparsers.add_parser("subcel", help='Experiments in subcellular localization', parents=[base_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_subcel.add_argument('--trainset',  help="Path to the trainset", default="data/Deeploc_seq/train.npz")
parser_subcel.add_argument('--testset',  help="Path to the testset", default="data/Deeploc_seq/test.npz")
parser_subcel.add_argument('--model', help="Choose which model you want to run", default="raw")
parser_subcel.add_argument('--n_features',  help="Embedding size if is_raw=True else number of features", type=int, default=20)
parser_subcel.add_argument('--n_filters',  help="Number of filters", type=int, default=20)
parser_subcel.add_argument('--in_dropout1d',  help="Input dropout feature", type=float, default=0.2)
parser_subcel.add_argument('--in_dropout2d',  help="Input dropout sequence", type=float, default=0.2)
parser_subcel.add_argument('--hid_dropout',  help="Hidden layers dropout", type=float, default=0.5)
parser_subcel.add_argument('--n_hid',  help="Number of hidden units", type=int, default=256)
parser_subcel.add_argument('--conv_kernels', nargs='+', help="Number of hidden units", default=[1,3,5,9,15,21])
parser_subcel.add_argument('--num_classes', help="Number of classes to predict from", type=int, default=10)
parser_subcel.add_argument('--att_size', help="Size of the attention", type=int, default=256)

# Secondary Structure Prediction
parser_secpred = subparsers.add_parser("secpred", help="Experiments in secondary structure prediction", parents=[base_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_secpred.add_argument('--trainset',  help="Path to the trainset", default="data/Deeploc_seq/train.npz")
parser_secpred.add_argument('--testset',  help="Path to the testset", default="data/Deeploc_seq/test.npz")
parser_secpred.add_argument('--model', help="Choose which model you want to run", default="something")
parser_secpred.add_argument('--crf', help="Turn on CRF", action="store_true")

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device
print("Arguments: ", args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
  torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

def results_save(fn):
  with open(fn, 'wb') as f:
    pickle.dump(results, f)

# Load data
print("Loading data...")
test_data = np.load(args.testset)
train_data = np.load(args.trainset)

# Test set
X_test = test_data['X_test']
y_test = test_data['y_test']
mask_test = test_data['mask_test']
mem_test = test_data['mem_test'].astype(np.int32)
unk_test = test_data['unk_test'].astype(np.int32)

# Training set
X_train = train_data['X_train']
y_train = train_data['y_train']
mask_train = train_data['mask_train']
partition = train_data['partition']
mem_train = train_data['mem_train']
unk_train = train_data['unk_train']

print("X_train.shape", X_train.shape)
print("X_test.shape", X_test.shape)

# Tokenize and remove invalid sequenzes
if (True): # is_raw
  X_train, mask = tokenize_sequence(X_train)

  X_train = np.asarray(X_train)
  y_train = y_train[mask]
  mask_train = mask_train[mask]
  partition = partition[mask]
  mem_train = mem_train[mask]
  unk_train = unk_train[mask]

  X_test, mask = tokenize_sequence(X_test)

  X_test = np.asarray(X_test)
  y_test = y_test[mask]
  mask_test = mask_test[mask]
  mem_test = mem_test[mask]
  unk_test = unk_test[mask]

print("Loading complete!") 

###############################################################################
# Training code
###############################################################################

# Training
results = ResultsContainer()
best_model = None
best_val_accs = []
best_val_models = []

Model = import_module('models.deeploc.{}'.format(args.model)).Model

for i in range(1,5):
  best_val_acc = 0
  best_val_model = None
  # Network compilation
  print("Compilation model {}".format(i))

  model = Model(args).to(device)
  print("Model: ", model)

  # Train and validation sets
  train_index = np.where(partition != i)
  val_index = np.where(partition == i)
  X_tr = X_train[train_index].astype(np.float32)
  X_val = X_train[val_index].astype(np.float32)
  y_tr = y_train[train_index].astype(np.int32)
  y_val = y_train[val_index].astype(np.int32)
  mask_tr = mask_train[train_index].astype(np.float32)
  mask_val = mask_train[val_index].astype(np.float32)
  mem_tr = mem_train[train_index].astype(np.int32)
  mem_val = mem_train[val_index].astype(np.int32)
  unk_tr = unk_train[train_index].astype(np.int32)
  unk_val = unk_train[val_index].astype(np.int32)

  print("Validation shape: {}".format(X_val.shape))
  print("Training shape: {}".format(X_tr.shape))
  
  for epoch in range(args.epochs):
    start_time = time.time()

    train_loss, confusion_train, confusion_mem_train = model.run_train(X_tr, y_tr, mask_tr, mem_tr, unk_tr)
    val_loss, confusion_valid, confusion_mem_valid, (alphas, targets, seq_lengths) = model.run_eval(X_val, y_val, mask_val, mem_val, unk_val)
    
    results.append_epoch(train_loss, val_loss, confusion_train.accuracy(), confusion_valid.accuracy()) 
    
    if confusion_valid.accuracy() > best_val_acc:
      best_val_acc = confusion_valid.accuracy()
      best_val_model = model

    if best_val_acc > results.best_val_acc:
      results.best_val_acc = best_val_acc
      best_model = model
    
    print('-' * 22, ' epoch: {:3d} / {:3d} - time: {:5.2f}s '.format(epoch, args.epochs-1, time.time() - start_time), '-' * 22 )
    print('| Train | loss {:.4f} | acc {:.2f}% | mem_acc {:.2f}% | Gorodkin {:2.2f} | IC {:2.2f}' 
          ' |'.format(train_loss, confusion_train.accuracy()*100, confusion_mem_train.accuracy()*100, gorodkin(confusion_train.ret_mat()), IC(confusion_train.ret_mat())))
    print('| Valid | loss {:.4f} | acc {:.2f}% | mem_acc {:.2f}% | Gorodkin {:2.2f} | IC {:2.2f}' 
          ' |'.format(val_loss, confusion_valid.accuracy()*100, confusion_mem_valid.accuracy()*100, gorodkin(confusion_valid.ret_mat()), IC(confusion_valid.ret_mat())))
    print('-' * 79)
    
    sys.stdout.flush()

  best_val_accs.append(best_val_acc)
  best_val_models.append(best_val_model)

for i, acc in enumerate(best_val_accs):
  print("Partion {:1d} : acc {:.2f}%".format(i, acc*100))

print("Average validation accuracy {:.2f}% \n".format((sum(best_val_accs)/len(best_val_accs))*100))

test_loss, confusion_test, confusion_mem_test, (alphas, targets, seq_lengths) = Model.run_test(args, best_val_models, X_test, y_test, mask_test, mem_test, unk_test)
print("ENSAMBLE TEST RESULTS")
print(confusion_test)
print(confusion_mem_test)
print("test accuracy:\t\t{:.2f} %".format(confusion_test.accuracy() * 100))
print("test mem accuracy:\t{:.2f} %".format(confusion_mem_test.accuracy() * 100))
print("test Gorodkin:\t\t{:.2f}".format(gorodkin(confusion_test.ret_mat())))
print("test IC:\t\t{:.2f}".format(IC(confusion_test.ret_mat())))

results.set_final(
  alph = alphas.cpu().detach().numpy(), 
  seq_len = seq_lengths.cpu().detach().numpy(), 
  targets = targets, 
  cf = confusion_test.ret_mat(),
  cf_mem = confusion_mem_test.ret_mat(), 
  acc = confusion_test.accuracy(), 
  acc_mem = confusion_mem_test.accuracy())

#model = best_model
#model_save(args.save)
#results_save(args.save_results)