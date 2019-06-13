import numpy as np
import random

train_data = np.load("data/Deeploc/train.npz")
y_train = train_data['y_train']
partition = train_data['partition']


def getUniform(num_targets):
    uniformPreds = []
    for i in range(0,num_targets):
        x = random.randint(0,9)
        uniformPreds.append(x)
    return uniformPreds

def computeAccuracy(targets, preds):
    assert len(targets) == len(preds)
    hits = 0
    for i in range(len(targets)):
        if targets[i] == preds[i]:
            hits += 1
    return hits/len(targets)*100

if __name__ == "__main__":
    print("Uniform Model")
    for i in range(1,5):
        train_index = np.where(partition != i)
        y_tr = y_train[train_index].astype(np.int32)
        print("Partition: ", i)
        preds = getUniform(len(y_tr))
        print("Accuracy: ", computeAccuracy(y_tr, preds))