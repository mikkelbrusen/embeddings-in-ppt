import numpy as np
import torch
from torch.autograd import Variable
import sys
sys.path.insert(0,'..')
import datautils.data as data
from models.seqpred_model import SeqPred 



data_gen = data.gen_data(num_iterations=5001, batch_size=64)
model = SeqPred()
for idx, batch in enumerate(data_gen.gen_train()):
    seq_lengths = batch['mask'].sum(1).astype(np.int32)
    #sort to be in decending order for pad packed to work
    perm_idx = np.argsort(-seq_lengths)
    seq_lengths = seq_lengths[perm_idx]
    inputs = batch['X'][perm_idx]
    inp = Variable(torch.from_numpy(inputs).type(torch.float))
    seq_lens = Variable(torch.from_numpy(seq_lengths).type(torch.int32)) 
    output = model(inp=inp, seq_lengths=seq_lens)
    preds = output.argmax(2)
    assert False
