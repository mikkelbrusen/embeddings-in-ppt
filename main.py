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

from utils import iterate_minibatches, ResultsContainer, tensor_to_onehot
from confusionmatrix import ConfusionMatrix
from metrics_mc import gorodkin, IC
from models.model import ABLSTM, StraightToLinear, SeqVec
from models.awd_model import AWD_Embedding
from datautils.dataloader import tokenize_sequence


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--trainset',  help="npz file with traning profiles data", default="data/Deeploc_seq/train.npz")
parser.add_argument('-t', '--testset',  help="npz file with test profiles data to calculate final accuracy", default="data/Deeploc_seq/test.npz")
parser.add_argument('-raw','--is_raw', help="Boolean telleing whether the sequences are raw (True) or profiles (False), default True", default=True)
parser.add_argument('-fe', '--n_features',  help="Embedding size if is_raw=True else number of features, default = 20", default=20)
parser.add_argument('-bs', '--batch_size',  help="Minibatch size, default = 128", default=128)
parser.add_argument('-e', '--epochs',  help="Number of training epochs, default = 300", default=300)
parser.add_argument('-n', '--n_filters',  help="Number of filters, default = 20", default=20)
parser.add_argument('-lr', '--learning_rate',  help="Learning rate, default = 0.0005", default=0.0005)
parser.add_argument('-id', '--in_dropout',  help="Input dropout, default = 0.2", default=0.2)
parser.add_argument('-hd', '--hid_dropout',  help="Hidden layers dropout, default = 0.5", default=0.5)
parser.add_argument('-hn', '--n_hid',  help="Number of hidden units, default = 256", default=256)
parser.add_argument('-cv', '--conv_sizes', nargs='+', help="Number of hidden units, default = [1,3,5,9,15,21]", default=[1,3,5,9,15,21])
parser.add_argument('-d', '--directions', help="Number of LSTM directions. 2 = bi-direcitonal, default = 2", default=2)
parser.add_argument('-att', '--att_size', help="Size of the attention, default = 256", default=256)
parser.add_argument('-ns', '--num_steps', help="Number of steps in attention, default = 10", default=10)
parser.add_argument('-ch', '--cell_hid_size', help="Number of hidden units in LSTMCell of multistep attention, default = 512", default=512)
parser.add_argument('-ms', '--is_multi_step', help="Indicate use of multi step attention, default = True", default=False)
parser.add_argument('-se', '--seed',  help="Seed for random number init., default = 123456", default=123456)
parser.add_argument('-clip', '--clip', help="Gradient clipping, default = 2", default=2)
current_time = time.strftime('%b_%d-%H_%M') # 'Oct_18-09:03'
parser.add_argument('-save', '--save', help="Path to best saved model, default = save/best_model_XXX.pt", default="save/best_model_" + current_time + ".pt")
parser.add_argument('-sr', '--save_results', help="Path to result object containing all kind of results, default = save/best_results_XXX.pt", default="save/best_results_" + current_time)
args = parser.parse_args()
print("Arguments: ", args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Input options
n_class = 10
batch_size = int(args.batch_size)
seq_len = 1000
n_hid = int(args.n_hid)
lr = float(args.learning_rate)
num_epochs = int(args.epochs)
drop_per = float(args.in_dropout)
drop_hid = float(args.hid_dropout)
n_filt = int(args.n_filters)
n_feat = int(args.n_features)
is_raw = args.is_raw
conv_sizes = args.conv_sizes
direcitons = int(args.directions)
att_size = int(args.att_size)
num_steps = int(args.num_steps)
cell_hid_size = int(args.cell_hid_size)
is_multi_step = args.is_multi_step

torch.manual_seed(args.seed)
np.random.seed(seed=int(args.seed))
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
if (is_raw):
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
def prepare_tensors(batch):
  inputs, targets, in_masks, targets_mem, unk_mem = batch

  seq_lengths = in_masks.sum(1).astype(np.int32)

  #sort to be in decending order for pad packed to work
  perm_idx = np.argsort(-seq_lengths)
  seq_lengths = seq_lengths[perm_idx]
  inputs = inputs[perm_idx]
  targets = targets[perm_idx]
  targets_mem = targets_mem[perm_idx]
  unk_mem = unk_mem[perm_idx]

  #convert to tensors
  if is_raw:
    inputs = Variable(torch.from_numpy(inputs)).type(torch.long).to(device) # (batch_size, seq_len)
  else:
    inputs = Variable(torch.from_numpy(inputs)).to(device) # (batch_size, seq_len, feature_size)

  seq_lengths = torch.from_numpy(seq_lengths).to(device)

  return inputs, seq_lengths, targets, targets_mem, unk_mem

def run_models(models, inputs, seq_lengths, train=True):
  with torch.no_grad():
    all_hidden, last_hidden, raw_all_hidden, dropped_all_hidden  = embed_model(input=inputs, hidden=hidden, seq_lengths=seq_lengths)

  #model_input =  last_hidden[-1][0].squeeze(0) # (bs, emb_size) Use only last hidden state

  #one_hot_inputs = tensor_to_onehot(inputs,n_feat+1).to(device)
  #one_hot_inputs = one_hot_inputs.permute(1,0,2)
  
  averaged_hidden = (dropped_all_hidden[0] + dropped_all_hidden[1]) #(seq_len, bs, 1280)
  #averaged_hidden = torch.cat((averaged_hidden, one_hot_inputs),dim=2)

  #model_input = all_hidden # (seq_len, bs, emb_size) Use all hidden states
  model_input = averaged_hidden #(seq_len, bs, emb_size)
  model_input = model_input.permute(1,0,2) # (bs, seq_len, emb_size) Use all hidden states

  optimizer.zero_grad()
  (output, output_mem), alphas = models[0](model_input, seq_lengths)
  
  #When multiple models are given, perform ensambling
  for i in range(1,len(models)):
    (out, out_mem), alphas  = models[i](model_input, seq_lengths)
    output = output + out
    output_mem = output_mem + out_mem

  #divide by number of models
  output = torch.div(output,len(models))
  output_mem = torch.div(output_mem,len(models))

  return output, output_mem, alphas

def calculate_loss_and_accuracy(output, output_mem, targets, targets_mem, unk_mem, confusion, confusion_mem):
  #Confusion Matrix
  preds = np.argmax(output.cpu().detach().numpy(), axis=-1)
  
  mem_preds = torch.round(output_mem).type(torch.int).cpu().detach().numpy()
  confusion.batch_add(targets, preds)
  confusion_mem.batch_add(targets_mem[np.where(unk_mem == 1)], mem_preds[np.where(unk_mem == 1)])
  
  unk_mem = Variable(torch.from_numpy(unk_mem)).type(torch.float).to(device)
  targets = Variable(torch.from_numpy(targets)).type(torch.long).to(device)
  targets_mem = Variable(torch.from_numpy(targets_mem)).type(torch.float).to(device)

  # squeeze from [batch_size,1] -> [batch_size] such that it matches weight matrix for BCE
  output_mem = output_mem.squeeze(1)
  targets_mem = targets_mem.squeeze(1)

  # calculate loss
  loss = criterion(input=output, target=targets)
  loss_mem = F.binary_cross_entropy(input=output_mem, target=targets_mem, weight=unk_mem, reduction="sum")
  loss_mem = loss_mem / sum(unk_mem)
  combined_loss = loss + 0.5 * loss_mem
  
  return combined_loss

def train(epoch):
  model.train()
  embed_model.train()

  train_err = 0
  train_batches = 0
  confusion_train = ConfusionMatrix(n_class)
  confusion_mem_train = ConfusionMatrix(num_classes=2)

  # Generate minibatches and train on each one of them	
  for batch in iterate_minibatches(X_tr, y_tr, mask_tr, mem_tr, unk_tr, batch_size):
    inputs, seq_lengths, targets, targets_mem, unk_mem = prepare_tensors(batch)
  
    output, output_mem, _ = run_models([model], inputs, seq_lengths, train=True)

    loss = calculate_loss_and_accuracy(output, output_mem, targets, targets_mem, unk_mem, confusion_train, confusion_mem_train)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()

    train_err += loss.item()
    train_batches += 1 

  train_loss = train_err / train_batches
  return train_loss, confusion_train, confusion_mem_train

def evaluate(x, y, mask, membranes, unks, models):
  embed_model.eval()
  for i in range(len(models)):
    models[i].eval()

  val_err = 0
  val_batches = 0
  confusion_valid = ConfusionMatrix(n_class)
  confusion_mem_valid = ConfusionMatrix(num_classes=2)

  with torch.no_grad():
    # Generate minibatches and train on each one of them	
    for batch in iterate_minibatches(x, y, mask, membranes, unks, batch_size, sort_len=False, shuffle=False, sample_last_batch=False):
      inputs, seq_lengths, targets, targets_mem, unk_mem = prepare_tensors(batch)

      output, output_mem, alphas = run_models(models, inputs, seq_lengths, train=False)

      loss = calculate_loss_and_accuracy(output, output_mem, targets, targets_mem, unk_mem, confusion_valid, confusion_mem_valid)

      val_err += loss.item()
      val_batches += 1

  val_loss = val_err / val_batches
  return val_loss, confusion_valid, confusion_mem_valid, (alphas, targets, seq_lengths)

# Training
results = ResultsContainer()
best_model = None
best_val_accs = []
best_val_models = []

# Initialize the AWD language model, and load in saved parameters
embed_model = AWD_Embedding(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True).to(device)
with open("awd_lstm/test_v2_statedict.pt", 'rb') as f:
		state_dict = torch.load(f, map_location='cuda' if torch.cuda.is_available() else 'cpu')
embed_model.load_state_dict(state_dict)

hidden = embed_model.init_hidden(batch_size)

for i in range(1,5):
  best_val_acc = 0
  best_val_model = None
  # Network compilation
  print("Compilation model {}".format(i))

  model = ABLSTM(batch_size, n_hid, n_feat, n_class, drop_per, drop_hid, n_filt, conv_kernel_sizes=conv_sizes, att_size=att_size, 
    cell_hid_size=cell_hid_size, num_steps=num_steps, directions=direcitons, is_multi_step=is_multi_step).to(device)
  #model = StraightToLinear(batch_size=batch_size, n_hid=320, n_class=n_class, drop_per=0.5).to(device)
  #model = SeqVec(batch_size=batch_size, inp_size=1280, n_hid=32, n_class=10, drop_per=0.25).to(device)

  print("Model: ", model)
  optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
	
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

  # Optimizers
  criterion = nn.CrossEntropyLoss()

  print("Validation shape: {}".format(X_val.shape))
  print("Training shape: {}".format(X_tr.shape))
	
  
  for epoch in range(num_epochs):
    start_time = time.time()

    train_loss, confusion_train, confusion_mem_train = train(epoch)
    val_loss, confusion_valid, confusion_mem_valid, (alphas, targets, seq_lengths) = evaluate(X_val, y_val, mask_val, mem_val, unk_val, [model])
    
    results.append_epoch(train_loss, val_loss, confusion_train.accuracy(), confusion_valid.accuracy()) 
    
    if confusion_valid.accuracy() > best_val_acc:
      best_val_acc = confusion_valid.accuracy()
      best_val_model = model

    if best_val_acc > results.best_val_acc:
      results.best_val_acc = best_val_acc
      best_model = model
    
    print('-' * 22, ' epoch: {:3d} / {:3d} - time: {:5.2f}s '.format(epoch, num_epochs, time.time() - start_time), '-' * 22 )
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

test_loss, confusion_test, confusion_mem_test, (alphas, targets, seq_lengths) = evaluate(X_test, y_test, mask_test, mem_test, unk_test, best_val_models)
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

model = best_model
model_save(args.save)
results_save(args.save_results)
