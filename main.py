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

from utils import iterate_minibatches, ResultsContainer
from confusionmatrix import ConfusionMatrix
from metrics_mc import gorodkin, IC
from model import ABLSTM


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--trainset',  help="npz file with traning profiles data", default="data/Deeploc/train.npz")
parser.add_argument('-t', '--testset',  help="npz file with test profiles data to calculate final accuracy", default="data/Deeploc/test.npz")
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
parser.add_argument('-ms', '--is_multi_step', help="Indicate use of multi step attention, default = True", default=True)
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
mem_test = test_data['mem_test']
unk_test = test_data['unk_test']

# Initialize utput vectors from test set
complete_alpha = np.zeros((X_test.shape[0],seq_len))
complete_context = np.zeros((X_test.shape[0],n_hid*2))
complete_test = np.zeros((X_test.shape[0],n_class))

# Training set
X_train = train_data['X_train']
y_train = train_data['y_train']
mask_train = train_data['mask_train']
partition = train_data['partition']
mem_train = train_data['mem_train']
unk_train = train_data['unk_train']

print(unk_train.shape)
print("Loading complete!")

# Number of features
n_feat = np.shape(X_test)[2]

###############################################################################
# Training code
###############################################################################

def evaluate(x,y,mask,membranes,unks):
  model.eval()
  val_err = 0
  val_batches = 0
  confusion_valid = ConfusionMatrix(n_class)

  with torch.no_grad():
    # Generate minibatches and train on each one of them	
    for batch in iterate_minibatches(x, y, mask, membranes, unks, batch_size, sort_len=False, shuffle=False, sample_last_batch=False):
      inputs, targets, in_masks, targets_mem, unk_mem = batch

      seq_lengths = in_masks.sum(1)
    
      #sort to be in decending order for pad packed to work
      perm_idx = np.argsort(-seq_lengths)
      seq_lengths = seq_lengths[perm_idx]
      inputs = inputs[perm_idx]
      targets = targets[perm_idx]
      targets_mem = targets_mem[perm_idx]
      unk_mem = unk_mem[perm_idx]

      #convert to tensors
      seq_lengths = torch.from_numpy(seq_lengths).to(device)
      inputs = torch.from_numpy(inputs).to(device)
      inputs = Variable(inputs)
      unk_mem = Variable(torch.from_numpy(unk_mem)).type(torch.float).to(device)

      #define criterion
      criterion_mem_val = nn.BCELoss(weight=unk_mem, reduction="sum")

      output, _ , alphas  = model(inputs, seq_lengths)

      preds = np.argmax(output.cpu().detach().numpy(), axis=-1)
      confusion_valid.batch_add(targets, preds)
      
      targets = Variable(torch.from_numpy(targets)).type(torch.long).to(device)
      targets_mem = Variable(torch.from_numpy(targets_mem)).type(torch.float).to(device)
      val_err += criterion(input=output, target=targets).item()
      val_batches += 1


  val_loss = val_err / val_batches
  val_accuracy = confusion_valid.accuracy()
  cf_val = confusion_valid.ret_mat()
  return val_loss, val_accuracy, cf_val, confusion_valid, (alphas, targets, seq_lengths)

def train():
  model.train()

  # Full pass training set
  train_err = 0
  train_batches = 0
  confusion_train = ConfusionMatrix(n_class)

  # Generate minibatches and train on each one of them	
  for batch in iterate_minibatches(X_tr, y_tr, mask_tr, mem_tr, unk_tr, batch_size, shuffle=True):
    inputs, targets, in_masks, targets_mem, unk_mem = batch
    seq_lengths = in_masks.sum(1)
    
    #sort to be in decending order for pad packed to work
    perm_idx = np.argsort(-seq_lengths)
    seq_lengths = seq_lengths[perm_idx]
    inputs = inputs[perm_idx]
    targets = targets[perm_idx]
    targets_mem = targets_mem[perm_idx]
    unk_mem = unk_mem[perm_idx]
    
    #convert to tensors
    seq_lengths = torch.from_numpy(seq_lengths).to(device)
    inputs = torch.from_numpy(inputs).to(device)
    inputs = Variable(inputs)
    unk_mem = Variable(torch.from_numpy(unk_mem)).type(torch.float).to(device)

    optimizer.zero_grad()
    (output, output_mem) , _ , _ = model(inputs, seq_lengths)
    np_targets = targets
    targets = Variable(torch.from_numpy(targets)).type(torch.long).to(device)
    targets_mem = Variable(torch.from_numpy(targets_mem)).type(torch.float).to(device)


    print(output_mem)
    print(targets_mem)
    loss = criterion(input=output, target=targets)
    loss_mem = F.binary_cross_entropy(input=output_mem, target=targets_mem) #criterion_mem_train(input=output_mem, target=targets_mem)
    
    print(loss_mem.shape)
    total_loss = loss + loss_mem
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()

    train_err += loss.item()
    train_batches += 1
    preds = np.argmax(output.cpu().detach().numpy(), axis=-1)
    confusion_train.batch_add(np_targets, preds)

  train_loss = train_err / train_batches
  train_accuracy = confusion_train.accuracy()
  cf_train = confusion_train.ret_mat()
  
  return train_loss, train_accuracy, cf_train

# Training
results = ResultsContainer()
best_model = None
best_val_accs = []


for i in range(1,5):
  best_val_acc = 0
  # Network compilation
  print("Compilation model {}".format(i))
  model = ABLSTM(batch_size, n_hid, n_feat, n_class, lr, drop_per, drop_hid, n_filt, conv_kernel_sizes=conv_sizes, att_size=att_size, 
    cell_hid_size=cell_hid_size, num_steps=num_steps, directions=direcitons, is_multi_step=is_multi_step).to(device)
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
    confusion_valid = ConfusionMatrix(n_class)
    train_loss, train_accuracy, cf_train = train()
    val_loss, val_accuracy, cf_val, confusion_valid, (alphas, targets, seq_lengths) = evaluate(X_val, y_val, mask_val, mem_val, mem_tr)
    
    results.loss_training.append(train_loss)
    results.loss_validation.append(val_loss)
    results.acc_training.append(train_accuracy)
    results.acc_validation.append(val_accuracy)
    results.epochs += 1
    
    if val_accuracy > best_val_acc:
      best_val_acc = val_accuracy

    if best_val_acc > results.best_val_acc:
      results.best_val_acc = best_val_acc
      results.best_cf_val = cf_val
      results.alphas = alphas.cpu().detach().numpy()
      results.targets = targets.cpu().detach().numpy()
      results.seq_lengths = seq_lengths.cpu().detach().numpy()
      best_model = model
    
    print('-' * 13, ' epoch: {:3d} / {:3d} - time: {:5.2f}s '.format(epoch, num_epochs, time.time() - start_time), '-' * 13 )

    print('| Train | loss {:.4f} | acc {:.2f}% | Gorodkin {:2.2f} | IC {:2.2f}' 
          ' |'.format(train_loss, train_accuracy*100, gorodkin(cf_train), IC(cf_train)))
    print('| Valid | loss {:.4f} | acc {:.2f}% | Gorodkin {:2.2f} | IC {:2.2f}' 
          ' |'.format(val_loss, val_accuracy*100, gorodkin(cf_val), IC(cf_val)))
    print('-' * 62)
    
    if epoch % 5 == 0 and epoch > 0:
      print(confusion_valid)
    
    sys.stdout.flush()

  best_val_accs.append(best_val_acc)

for i, acc in enumerate(best_val_accs):
  print("Partion {:1d} : acc {:.2f}%".format(i, acc*100))

print("Average accuracy {:.2f}%".format((sum(best_val_accs)/len(best_val_accs))*100))

model = best_model
model_save(args.save)
results_save(args.save_results)

test_loss, test_accuracy, cf_test, confusion_test, _ = evaluate(X_test, y_test, mask_test, mem_test, unk_test)
print("FINAL TEST RESULTS")
print(confusion_test)
print("test accuracy:\t\t{:.2f} %".format(test_accuracy * 100))
print("test Gorodkin:\t\t{:.2f}".format(gorodkin(cf_test)))
print("test IC:\t\t{:.2f}".format(IC(cf_test)))