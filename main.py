import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import os
import time
import argparse 
import sys
from importlib import import_module

parser = argparse.ArgumentParser()
# Base parser for all
base_parser = argparse.ArgumentParser(add_help=False)
base_parser.add_argument('--batch_size',  help="Minibatch size", type=int, default=128)
base_parser.add_argument('--epochs',  help="Number of training epochs", type=int, default=200)
base_parser.add_argument('--learning_rate',  help="Learning rate", type=float, default=0.0005)
base_parser.add_argument('--optimizer',  help="optimizer", default="adam")
base_parser.add_argument('--seed',  help="Seed for random number init.", type=int, default=123456)
base_parser.add_argument('--clip', help="Gradient clipping", type=float, default=2.0)
base_parser.add_argument('--n_features',  help="Embedding size or number of features in profiles", type=int, default=20)
base_parser.add_argument('--n_filters',  help="Number of filters", type=int, default=20)
base_parser.add_argument('--in_dropout1d',  help="Input dropout feature", type=float, default=0.2)
base_parser.add_argument('--in_dropout2d',  help="Input dropout sequence", type=float, default=0.2)
base_parser.add_argument('--hid_dropout',  help="Hidden layers dropout", type=float, default=0.5)
base_parser.add_argument('--n_hid',  help="Number of hidden units", type=int, default=256)
base_parser.add_argument('--conv_kernels', nargs='+', help="Number of hidden units", default=[1,3,5,9,15,21])
base_parser.add_argument('--num_classes', help="Number of classes to predict from", type=int, default=10)
base_parser.add_argument('--do_testing', help="Run best model(s) on test data", action="store_true")
base_parser.add_argument('--test_only', help="Run best model(s) on test data only", action="store_true")

### SUBPARSERS ### 
subparsers = parser.add_subparsers(dest='parser_name')

# Subcellular
parser_subcel = subparsers.add_parser("subcel", help='Experiments in subcellular localization', parents=[base_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_subcel.add_argument('--config', help="Choose which config you want to run", default="deeploc_raw")
parser_subcel.add_argument('--att_size', help="Size of the attention", type=int, default=256)

# Secondary Structure Prediction
parser_secpred = subparsers.add_parser("secpred", help="Experiments in secondary structure prediction", parents=[base_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_secpred.add_argument('--trainset',  help="Path to the trainset", default="data/SecPred/train_no_x.npy")
parser_secpred.add_argument('--testset',  help="Path to the testset", default="data/SecPred/test_no_x.npy")
parser_secpred.add_argument('--config', help="Choose which model you want to run", default="soenderby_raw")
parser_secpred.add_argument('--crf', help="Turn on CRF", action="store_true")
parser_secpred.add_argument('--cb513', help="Use CB513 as test set", action="store_true")
parser_secpred.add_argument('--raw', help="Use raw sequence data", action="store_true")
parser_secpred.add_argument('--n_l1',  help="Size of first linear layer", type=int, default=500)
parser_secpred.add_argument('--n_l2',  help="Size of second linear layer", type=int, default=400)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device

current_time = time.strftime('%b_%d-%H_%M') # 'Oct_18-09:03'
args.start_time = current_time

print("#"*40)
print("#", ' '*36, "#")
print("#", '{:^36}'.format(args.parser_name + " - " + args.config), "#")
print("#", ' '*36, "#")
print("#"*40)

print("Arguments: ", args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
  torch.cuda.manual_seed(args.seed)

###############################################################################
# Training
###############################################################################
Config = import_module('configs.{}.{}'.format(args.parser_name, args.config)).Config
config = Config(args)

if not args.test_only:
  config.trainer()

###############################################################################
# Testing
###############################################################################
if args.do_testing or args.test_only:
  config.tester()