import numpy as np
import os
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

def tokenize_sequence(data, seq_len=1000):
    amino_dictionary = Dictionary()
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    illegal = set('BXZU')

    for letter in alphabet:
        amino_dictionary.add_word(letter)

    data_tokenized = []
    mask_legal = []
    illegals = 0
    for i, entry in enumerate(data):
        sequence = np.empty(seq_len, dtype=int)
        sequence[:] = 20
        
        # Check if any illegal amino acids
        if any((c in illegal) for c in entry):
            illegals += 1
            continue
        
        #Add index of valid protein and start tokenizing
        mask_legal.append(i)

        for j, char in enumerate(entry.split('*')[0]):
            sequence[j] = amino_dictionary.word2idx[char]
        data_tokenized.append(sequence) 
    print("Number of Illegals: ", illegals)
    return data_tokenized, mask_legal



if __name__ == "__main__":
    #train_data = np.load("data/Deeploc_seq/train.npz")
    #X_train = train_data['X_train']
    #y_train = train_data['y_train']

    # First is illegal
    X_train = ["ABCD","EFGH","IKLM","NP**"]
    y_train = np.array([1,2,3,4])

    X_train, mask = tokenize_sequence(X_train,4)
    mask = np.asarray(mask)
    y_train = y_train[mask]

    print("X_train: ", X_train)
    print("y_train: ", y_train)
    
