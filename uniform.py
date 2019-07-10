import numpy as np
import random
import sys
import dataloaders.secpred as data
subcell = False
if subcell:
    train_data = np.load("data/Deeploc/train.npz")
    y_train = train_data['y_train']
    partition = train_data['partition']   
else:
    data_dict, _ = data.load_data(is_cb513=True, is_raw=True, train_path="data/SecPred/train_raw.npz", test_path="data/SecPred/test_raw.npz")
    y_tr = data_dict["t_test"].astype(np.int32)
    seq_len = data_dict["length_test"].astype(np.int32)



def getUniform(num_targets, num_classes):
    uniformPreds = []
    for i in range(0,num_targets):
        if subcell:
            x = random.randint(0,num_classes-1)
        else:
            x = np.random.randint(num_classes,size=seq_len[i])       
        uniformPreds.append(x)
    return uniformPreds

def computeAccuracy(targets, preds):
    assert len(targets) == len(preds)
    hits = 0
    if subcell:
        for i in range(len(targets)):
            if targets[i] == preds[i]:
                hits += 1
        return hits/len(targets)*100
    else:
        total = 0
        for i in range(len(preds)):
            for j in range(len(preds[i])):
                total += 1
                if targets[i][j] == preds[i][j]:
                    hits += 1
        return hits/total*100

if __name__ == "__main__":
    if subcell:
        print("Uniform Model")
        for i in range(1,5):
            train_index = np.where(partition != i)
            y_tr = y_train[train_index].astype(np.int32)
            print("Partition: ", i)
            preds = getUniform(num_targets=len(y_tr), num_classes=10)
            print("Accuracy: ", computeAccuracy(y_tr, preds))
    else:
        print("Uniform Model")
        preds = getUniform(num_targets=len(y_tr), num_classes=8)
        print("Accuracy: ", computeAccuracy(y_tr, preds))


    