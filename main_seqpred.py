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

num_epochs = 5
clip_norm = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate():
    eval_err = 0
    eval_batches = 0
    accuracy = 0
    model.eval()
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
            mask = Variable(torch.from_numpy(mask).type(torch.float)).to(device)
            targets = Variable(torch.from_numpy(targets).type(torch.long)).to(device)

            output = model(inp=inp, seq_lengths=seq_lens)
        
            # calculate loss
            loss = 0
            loss_preds = output.permute(1,0,2)
            loss_mask = mask.permute(1,0)
            loss_targets = targets.permute(1,0)
            for i in range(loss_preds.size(0)):
                loss += sum(F.nll_loss(loss_preds[i], loss_targets[i], reduction='none') * loss_mask[i])/sum(loss_mask[i])

            accuracy += calculate_accuracy(preds=output, targets=targets, mask=mask)

            eval_err += loss.item()
            eval_batches += 1

    total_accuracy = accuracy / eval_batches
    eval_loss = eval_err / eval_batches
    return eval_loss, total_accuracy


def train():
    train_err = 0
    train_batches = 0
    accuracy = 0
    model.train()
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
        #loss = 0
        loss_preds = output.permute(1,0,2)
        loss_mask = mask.permute(1,0)
        loss_targets = targets.permute(1,0)
        #for i in range(loss_preds.size(0)):
        #    loss += sum(F.nll_loss(loss_preds[i], loss_targets[i], reduction='none') * loss_mask[i])/sum(loss_mask[i])
        loss = crf(loss_preds, loss_targets, loss_mask)
        loss.backward()

        preds = torch.tensor(crf.decode(emissions=loss_preds), dtype=torch.int64)
        accuracy += calculate_accuracy(preds=preds, targets=targets, mask=mask)

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        train_err += loss.item()
        train_batches += 1

        print("Batch: ", idx)
        sys.stdout.flush()

    total_accuracy = accuracy / train_batches
    train_loss = train_err / train_batches
    return train_loss, total_accuracy

def calculate_accuracy(preds, targets, mask):
    #preds = preds.argmax(2).type(torch.float)
    correct = preds.type(torch.float).eq(targets.type(torch.float)).type(torch.float) * mask.type(torch.float)
    return torch.sum(correct) / torch.sum(mask)
    

# Network compilation
model = SeqPred().to(device)
print("Model: ", model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
data_gen = data.gen_data(num_iterations=2, batch_size=64)
crf = CRF(num_tags=8)
for epoch in range(num_epochs):
    start_time = time.time()

    train_loss, train_accuracy = train()
    val_loss, val_accuracy = evaluate()

    print('-' * 22, ' epoch: {:3d} / {:3d} - time: {:5.2f}s '.format(epoch, num_epochs, time.time() - start_time), '-' * 22 )
    print('| Train | loss {:.4f} | acc {:.2f}%' 
            ' |'.format(train_loss, train_accuracy*100))
    print('| Valid | loss {:.4f} | acc {:.2f}% ' 
            ' |'.format(val_loss, val_accuracy*100))
    print('-' * 79)

    sys.stdout.flush()
