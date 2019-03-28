import numpy as np
import torch
import os

from Bio import SeqIO
from collections import Counter

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


train = []
test = []
train_targets = []
train_masks = []
train_mem_targets = []
train_unk = []
test_masks = []
test_targets = []
test_mem_targets = []
test_unk = []

amino_dictionary = Dictionary()
target_dictionary = Dictionary()

alphabet = 'ABCDEFGHIKLMNPQRSTUVWXYZ'
classes = ['Nucleus','Cytoplasm','Extracellular','Mitochondrion','Cell.membrane','Endoplasmic.reticulum','Plastid','Golgi.apparatus', 'Lysosome/Vacuole', 'Peroxisome']

for letter in alphabet:
    amino_dictionary.add_word(letter)
for location in classes:
    target_dictionary.add_word(location)

for seq_record in SeqIO.parse("data/Deeploc_raw/deeploc_data.fasta", "fasta"):
    sequence = np.zeros(1000, dtype=int)
    mask = np.zeros(1000, dtype=int)
    seq_desc = seq_record.description.split()
    seq_targets = seq_desc[1].split('-')

    if (seq_targets[0] == 'Cytoplasm/Nucleus'):
        continue
    
    #Sequence
    if (len(seq_record.seq) >= 1000):
        left = seq_record.seq[:500]
        right = seq_record.seq[-500:]
        seq = left + right

        for j,letter in enumerate(seq):
            sequence[j] = amino_dictionary.word2idx[letter]
            mask[j] = 1

    else:
       for j,letter in enumerate(seq_record.seq):
            sequence[j] = amino_dictionary.word2idx[letter]
            mask[j] = 1

    #Targets
    unk = 0
    mem_target = 0

    target = target_dictionary.word2idx[seq_targets[0]]
    if (seq_targets[1] == 'M'):
        mem_target = 1
        unk = 1
    elif (seq_targets[1] == 'S'):
        unk = 1

    #
    if (len(seq_desc) == 3):
        test.append(sequence)
        test_masks.append(mask)
        test_targets.append(target)
        test_mem_targets.append(mem_target)
        test_unk.append(unk)
    else:
        train.append(sequence)
        train_masks.append(mask)
        train_targets.append(target)
        train_mem_targets.append(mem_target)
        train_unk.append(unk)
    
#print(len(train) + len(test))
#print(len(test))
print(train_mem_targets)
print(train_unk)
    




