import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import time

sys.path.insert(0,'..')
import datautils.data as data
from models.seqpred_model import SeqPred

clip_norm = 1
num_batch = 44
batch_size = 128
num_epochs = 100
lr = 1e-3
number_outputs = 8
crf_on = False
if crf_on:
    from torchcrf import CRF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(crf_on, is_test):
    accuracy = 0
    model.eval()
    if crf_on:
        crf.eval()
    with torch.no_grad():
        if is_test:
            inputs, targets, mask, seq_lengths = data_gen.get_test_data()
        else:
            inputs, targets, mask, seq_lengths = data_gen.get_valid_data()

        #sort to be in decending order for pad packed to work
        perm_idx = np.argsort(-seq_lengths)
        seq_lengths = seq_lengths[perm_idx]
        inputs = inputs[perm_idx]
        targets = targets[perm_idx]
        mask = mask[perm_idx]
        inp = Variable(torch.from_numpy(inputs).type(torch.float)).to(device)
        seq_lens = Variable(torch.from_numpy(seq_lengths).type(torch.int32)).to(device)
        mask_byte = Variable(torch.from_numpy(mask).type(torch.ByteTensor)).to(device)
        mask_float = Variable(torch.from_numpy(mask).type(torch.float)).to(device)
        targets = Variable(torch.from_numpy(targets).type(torch.long)).to(device)
        num_samples = inp.size(0)

        output = model(inp=inp, seq_lengths=seq_lens)
        #outputs = torch.cat(output, axis=0)
                #seq_lengths ...
        
        if crf_on:
            # calculate loss
            loss = -crf(output, targets, mask_byte)

            # calculate accuaracy
            preds_list = crf.decode(emissions=output, mask=mask_byte)
            accuracy += calculate_accuracy_crf(preds=preds_list, targets=targets, mask=mask_byte)
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
            accuracy += calculate_accuracy(preds=output, targets=targets, mask=mask_float)

        return loss, accuracy


def train(crf_on, num_batch):
    train_err = 0
    total_samples = 0
    accuracy = 0
    train_batches = 0
    model.train()
    if crf_on:
        crf.train()
    start_time = time.time()
    for b in range(num_batch):
        batch = next(data_gen_train)
        seq_lengths = batch['length']
        #sort to be in decending order for pad packed to work
        perm_idx = np.argsort(-seq_lengths)
        seq_lengths = seq_lengths[perm_idx]
        inputs = batch['X'][perm_idx]
        targets = batch['t'][perm_idx]
        mask = batch['mask'][perm_idx]
        inp = Variable(torch.from_numpy(inputs).type(torch.float)).to(device)
        seq_lens = Variable(torch.from_numpy(seq_lengths).type(torch.int32)).to(device)
        mask_byte = Variable(torch.from_numpy(mask).type(torch.ByteTensor)).to(device)
        mask_float = Variable(torch.from_numpy(mask).type(torch.float)).to(device)
        targets = Variable(torch.from_numpy(targets).type(torch.long)).to(device)
        num_samples = inp.size(0)

        optimizer.zero_grad()
        output = model(inp=inp, seq_lengths=seq_lens)
        
        if crf_on:
            # calculate loss
            loss = -crf(output, targets, mask_byte)
            loss.backward()

            # calculate accuaracy
            preds_list = crf.decode(emissions=output, mask=mask_byte)
            accuracy += calculate_accuracy_crf(preds=preds_list, targets=targets, mask=mask_byte)

            torch.nn.utils.clip_grad_norm_(parameters=list(model.parameters()) + list(crf.parameters()), max_norm=clip_norm)
            #torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=clip_norm)
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
            accuracy += calculate_accuracy(preds=output, targets=targets, mask=mask_float)

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=clip_norm)
        

        optimizer.step()

        train_err += loss.item()
        train_batches += 1
    train_accuracy = accuracy / train_batches
    train_loss = train_err / train_batches
    return train_loss, train_accuracy

def calculate_accuracy_crf(preds, targets, mask):
    correct = 0
    for i in range(len(preds)):
        pred = torch.tensor(preds[i]).type(torch.float).to(device)
        target = targets[i][mask[i]].type(torch.float).to(device)
        correct += torch.sum(pred.eq(target))
    return correct.type(torch.float) / torch.sum(mask.type(torch.float))

def calculate_accuracy(preds, targets, mask):
    preds = preds.argmax(2).type(torch.float)
    correct = preds.type(torch.float).eq(targets.type(torch.float)).type(torch.float) * mask.type(torch.float)
    return torch.sum(correct) / torch.sum(mask)

#def proteins_acc(out, label, mask):
    #    out = np.argmax(out, axis=2)
    #    return np.sum(((out == label).flatten()*mask.flatten())).astype('float32') / np.sum(mask).astype('float32')

# Network compilation
model = SeqPred().to(device)
best_model = model
print("batch_size", batch_size)
print("num_batch", num_batch)
print("clip_norm", clip_norm)
print("lr", lr)
print("Model: ", model)
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
if crf_on:
    crf = CRF(num_tags=number_outputs, batch_first=True).to(device)
    best_crf = crf
    print("CRF: ", crf)
    optimizer = torch.optim.Adam(params=list(model.parameters()) + list(crf.parameters()), lr=lr)

data_gen = data.gen_data(num_iterations=num_batch, batch_size=batch_size)
data_gen_train = data_gen.gen_train()

#for epoch in range(num_epochs):
#    for b in range(num_batch):
#        batch = next(data_gen_train)
#        print(b)

best_val_acc = 0.0
for epoch in range(num_epochs):
    start_time = time.time()
    train_loss, train_accuracy = train(crf_on=crf_on, num_batch=num_batch)
    val_loss, val_accuracy = evaluate(crf_on=crf_on, is_test=False)
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        idx = epoch
        if crf_on:
            best_crf = crf
        best_model = model


    print('-' * 22, ' epoch: {:3d} / {:3d} - time: {:5.2f}s '.format(epoch, num_epochs, time.time() - start_time), '-' * 22 )
    #Train
    print('| Train | loss {:.4f} | acc {:.2f}%'
    ' |'.format(train_loss, train_accuracy*100))
    print('| Valid | loss {:.4f} | acc {:.2f}%' 
    ' |'.format(val_loss, val_accuracy*100))
    print('-' * 79)
    sys.stdout.flush()

print("BEST RESULTS")
print('| Valid | epoch {:3d} | acc {:.2f}% ' 
            ' |'.format(idx, best_val_acc*100))
print('-' * 79)

#test
model = best_model
if crf_on:
    crf = best_crf
test_loss, test_accuracy = evaluate(crf_on=crf_on, is_test=True)

print('| Test | acc {:.2f}% ' 
            ' |'.format(test_accuracy*100))
print('-' * 79)