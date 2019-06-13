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

parser = argparse.ArgumentParser()
# Base parser for all
base_parser = argparse.ArgumentParser(add_help=False)
base_parser.add_argument('--batch_size',  help="Minibatch size", type=int, default=128)
base_parser.add_argument('--epochs',  help="Number of training epochs", type=int, default=200)
base_parser.add_argument('--learning_rate',  help="Learning rate", type=float, default=0.0005)
base_parser.add_argument('--seed',  help="Seed for random number init.", type=int, default=123456)
base_parser.add_argument('--clip', help="Gradient clipping", type=int, default=2)
base_parser.add_argument('--do_testing', help="Run best model(s) on test data", action="store_true")
current_time = time.strftime('%b_%d-%H_%M') # 'Oct_18-09:03'
base_parser.add_argument('--save', help="Path to best saved model", default="save/best_model_" + current_time + ".pt")
base_parser.add_argument('--save_results', help="Path to result object containing all kind of results", default="save/best_results_" + current_time)

### SUBPARSERS ### 
subparsers = parser.add_subparsers(dest='parser_name')

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
#parser_secpred.add_argument('--trainset',  help="Path to the trainset", default="data/Deeploc_seq/train.npz")
#parser_secpred.add_argument('--testset',  help="Path to the testset", default="data/Deeploc_seq/test.npz")
parser_secpred.add_argument('--model', help="Choose which model you want to run", default="secPred")
parser_secpred.add_argument('--crf', help="Turn on CRF", action="store_true")
parser_secpred.add_argument('--cb513', help="Use CB513 as test set", action="store_true")
parser_secpred.add_argument('--input_size',  help="Size of input", type=int, default=42)
parser_secpred.add_argument('--n_l1',  help="Size of first linear layer", type=int, default=500)
parser_secpred.add_argument('--n_rnn_hid',  help="Number of hidden units in rnn", type=int, default=500)
parser_secpred.add_argument('--n_l2',  help="Size of second linear layer", type=int, default=400)
parser_secpred.add_argument('--n_outputs',  help="Number of outputs", type=int, default=8)


args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device
print("Arguments: ", args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
  torch.cuda.manual_seed(args.seed)

###############################################################################
# Training
###############################################################################
Config = import_module('models.{}.{}'.format(args.parser_name, args.model)).Config
config = Config(args)

best_val_accs, best_val_models = config.trainer()

###############################################################################
# Testing
###############################################################################
if args.do_testing:
  config.tester(best_val_models)