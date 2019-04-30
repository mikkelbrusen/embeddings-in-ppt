import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import time
from torchcrf import CRF
sys.path.insert(0,'..')
import datautils.data as data
from models.seqpred_model import SeqPred 

clip_norm = 1
valid_every = 50
num_iterations = 5001
lr = 0.001
batch_size = 64
number_outputs = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate():
    eval_err = 0
    eval_batches = 0
    accuracy = 0
    model.eval()
    crf.eval()
    with torch.no_grad():
        for idx, batch in enumerate(data_gen.gen_valid()):
            seq_lengths = batch['mask'].sum(1).astype(np.int32)
            #sort to be in decending order for pad packed to work
            perm_idx = np.argsort(-seq_lengths)
            seq_lengths = seq_lengths[perm_idx]
            inputs = batch['X'][perm_idx]
            targets = batch['t'][perm_idx]
            mask = batch['mask'][perm_idx]
            inp = Variable(torch.from_numpy(inputs).type(torch.float)).to(device)
            seq_lens = Variable(torch.from_numpy(seq_lengths).type(torch.int32)).to(device)
            mask = Variable(torch.from_numpy(mask).type(torch.ByteTensor)).to(device)
            targets = Variable(torch.from_numpy(targets).type(torch.long)).to(device)

            output = model(inp=inp, seq_lengths=seq_lens)
        
            # calculate loss
            loss = -crf(output, targets, mask)

            preds_list = crf.decode(emissions=output, mask=mask)
            accuracy += calculate_accuracy(preds=preds_list, targets=targets, mask=mask)

            eval_err += loss.item()
            eval_batches += 1

    total_accuracy = accuracy / eval_batches
    eval_loss = eval_err / eval_batches
    return eval_loss, total_accuracy


def train():
    train_err = 0
    train_batches = 0
    accuracy = 0
    best_val_acc = 0
    best_iteration = 0
    model.train()
    crf.train()
    start_time = time.time()
    for idx, batch in enumerate(data_gen.gen_train()):
        seq_lengths = batch['mask'].sum(1).astype(np.int32)
        #sort to be in decending order for pad packed to work
        perm_idx = np.argsort(-seq_lengths)
        seq_lengths = seq_lengths[perm_idx]
        inputs = batch['X'][perm_idx]
        targets = batch['t'][perm_idx]
        mask = batch['mask'][perm_idx]
        inp = Variable(torch.from_numpy(inputs).type(torch.float)).to(device)
        seq_lens = Variable(torch.from_numpy(seq_lengths).type(torch.int32)).to(device)
        mask = Variable(torch.from_numpy(mask).type(torch.ByteTensor)).to(device)
        targets = Variable(torch.from_numpy(targets).type(torch.long)).to(device)

        optimizer.zero_grad()
        output = model(inp=inp, seq_lengths=seq_lens)
        
        # calculate loss
        loss = -crf(output, targets, mask)
        loss.backward()

        # calculate accuaracy
        preds_list = crf.decode(emissions=output, mask=mask)
        accuracy += calculate_accuracy(preds=preds_list, targets=targets, mask=mask)

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        train_err += loss.item()
        train_batches += 1

        if (idx % valid_every) == 0:
            print('-' * 22, ' iteration: {:3d} / {:3d} - time: {:5.2f}s '.format(idx, num_iterations, time.time() - start_time), '-' * 22 )
			#Train
            total_accuracy = accuracy / train_batches
            train_loss = train_err / train_batches
            print('| Train | loss {:.4f} | acc {:.2f}%' 
            ' |'.format(train_loss, total_accuracy*100))

			#evaluate
            val_loss, val_accuracy = evaluate()

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_iteration = idx

            print('| Valid | loss {:.4f} | acc {:.2f}% ' 
            ' |'.format(val_loss, val_accuracy*100))
            print('-' * 79)
            model.train()
            crf.train()
            accuracy = 0
            train_batches = 0
            train_err = 0
            start_time = time.time()
            sys.stdout.flush()
    
    return best_val_acc, best_iteration

def calculate_accuracy(preds, targets, mask):
    correct = 0
    for i in range(len(preds)):
        pred = torch.tensor(preds[i]).type(torch.float).to(device)
        target = targets[i][mask[i]].type(torch.float).to(device)
        correct += torch.sum(pred.eq(target))

    return correct.type(torch.float) / torch.sum(mask.type(torch.float))
    

# Network compilation
model = SeqPred().to(device)
print("Model: ", model)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
data_gen = data.gen_data(num_iterations=num_iterations, batch_size=batch_size)
crf = CRF(num_tags=number_outputs, batch_first=True).to(device)
best_acc, idx = train()
print("BEST RESULTS")
print('| Valid | iteration {:3d} | acc {:.2f}% ' 
            ' |'.format(idx, best_acc*100))
print('-' * 79)
