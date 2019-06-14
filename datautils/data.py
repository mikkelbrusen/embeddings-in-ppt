import numpy as np
import os.path
import subprocess
import sys
import copy
sys.path.insert(0,'..')
import datautils.casphandle as casphandle
import utils

TRAIN_PATH_CULLPDB = 'data/SecPred/train_nf.npy'
TRAIN_PATH_CB513 = 'data/SecPred/train.npy'
TEST_PATH = 'data/SecPred/test.npy'
SAVE_DATASETS = True
##### TRAIN DATA #####

def get_train_cb513(seq_len=None):
  if not os.path.isfile(TRAIN_PATH_CB513):
    print("Train path is not downloaded ...")
    subprocess.call("./download_train.sh", shell=True)
  else:
    print("Train path is downloaded ...")
  print("Loading train data ...")
  X_in = np.load(TRAIN_PATH_CB513)
  X = np.reshape(X_in,(5534,700,57))
  Y = copy.deepcopy(X)
  del X_in
  X = X[:,:,:]
  labels = X[:,:,22:30]
  mask = X[:,:,30] * -1 + 1

  a = np.arange(0,21)
  b = np.arange(35,56)
  c = np.hstack((a,b))

  X = X[:,:,c]

  

  # getting meta
  num_seqs = np.size(X,0)
  seqlen = np.size(X,1)
  d = np.size(X,2)
  num_classes = 8

  #### REMAKING LABELS ####
  X = X.astype("float32")
  mask = mask.astype("float32")
  # Dummy -> concat
  vals = np.arange(0,8)
  labels_new = np.zeros((num_seqs,seqlen))
  for i in range(np.size(labels,axis=0)):
    labels_new[i,:] = np.dot(labels[i,:,:], vals)
  labels_new = labels_new.astype('int32')
  labels = labels_new

  print("Loading splits ...")
  ##### SPLITS #####
  # getting splits (cannot run before splits are made)
  #split = np.load("data/split.pkl")

  seq_names = np.arange(0,num_seqs)
  #np.random.shuffle(seq_names)

  X_train = X[seq_names[0:5278]]
  X_valid = X[seq_names[5278:5534]]
  labels_train = labels[seq_names[0:5278]]
  labels_valid = labels[seq_names[5278:5534]]
  mask_train = mask[seq_names[0:5278]]
  mask_valid = mask[seq_names[5278:5534]]
  num_seq_train = np.size(X_train,0)
  num_seq_valid = np.size(X_valid,0)
  if seq_len is not None:
    X_train = X_train[:, :seq_len]
    X_valid = X_valid[:, :seq_len]
    labels_train = labels_train[:, :seq_len]
    labels_valid = labels_valid[:, :seq_len]
    mask_train = mask_train[:, :seq_len]
    mask_valid = mask_valid[:, :seq_len]
  len_train = np.sum(mask_train, axis=1)
  len_valid = np.sum(mask_valid, axis=1)

  if SAVE_DATASETS:
    save_raw_dataset(data=Y[:,:,np.arange(0,22)], Y=Y, masks=mask, targets=labels, is_test=False)

  return X_train, X_valid, labels_train, labels_valid, mask_train, \
      mask_valid, len_train, len_valid, num_seq_train

def get_train_cullpdb(seq_len=None):
  if not os.path.isfile(TRAIN_PATH_CULLPDB):
    print("Train path is not downloaded ...")
    subprocess.call("./download_train.sh", shell=True)
  else:
    print("Train path is downloaded ...")
  print("Loading train data ...")
  X_in = np.load(TRAIN_PATH_CULLPDB)
  X = np.reshape(X_in,(6133,700,57))
  Y = copy.deepcopy(X)
  del X_in
  X = X[:,:,:]
  labels = X[:,:,22:30]
  mask = X[:,:,30] * -1 + 1

  a = np.arange(0,21)
  b = np.arange(35,56)
  c = np.hstack((a,b))

  X = X[:,:,c]

  # getting meta
  num_seqs = np.size(X,0)
  seqlen = np.size(X,1)
  d = np.size(X,2)
  num_classes = 8

  #### REMAKING LABELS ####
  X = X.astype("float32")
  mask = mask.astype("float32")
  # Dummy -> concat
  vals = np.arange(0,8)
  labels_new = np.zeros((num_seqs,seqlen))
  for i in range(np.size(labels,axis=0)):
    labels_new[i,:] = np.dot(labels[i,:,:], vals)
  labels_new = labels_new.astype('int32')
  labels = labels_new

  print("Loading splits ...")
  ##### SPLITS #####
  # getting splits (cannot run before splits are made)
  #split = np.load("data/split.pkl")

  seq_names = np.arange(0,num_seqs)
  #np.random.shuffle(seq_names)

  X_train = X[seq_names[0:5600]]
  X_valid = X[seq_names[5877:6133]]
  X_test = X[seq_names[5605:5877]]
  labels_train = labels[seq_names[0:5600]]
  labels_valid = labels[seq_names[5877:6133]]
  labels_test = labels[seq_names[5605:5877]]
  mask_train = mask[seq_names[0:5600]]
  mask_valid = mask[seq_names[5877:6133]]
  mask_test = mask[seq_names[5605:5877]]
  num_seq_train = np.size(X_train,0)
  num_seq_valid = np.size(X_valid,0)
  len_train = np.sum(mask_train, axis=1)
  len_valid = np.sum(mask_valid, axis=1)
  len_test = np.sum(mask_test, axis=1)

  if SAVE_DATASETS:
    save_raw_dataset(data=Y[:,:,np.arange(0,22)], Y=Y, masks=mask, targets=labels, is_test=False, is_cullpdb=True)

  return X_train, X_valid, X_test, labels_train, labels_valid, labels_test, mask_train, \
      mask_valid, mask_test, len_train, len_valid, len_test, num_seq_train
#del split
##### TEST DATA #####

def get_test(seq_len=None):
  if not os.path.isfile(TEST_PATH):
    subprocess.call("./download_test.sh", shell=True)
  print("Loading test data ...")
  X_test_in = np.load(TEST_PATH)
  X_test = np.reshape(X_test_in,(514,700,57))
  Y_test = copy.deepcopy(X_test)
  del X_test_in
  X_test = X_test[:,:,:].astype("float32")
  labels_test = X_test[:,:,22:30].astype('int32')
  mask_test = X_test[:,:,30].astype("float32") * -1 + 1

  a = np.arange(0,21)
  b = np.arange(35,56)
  c = np.hstack((a,b))
    
  X_test = X_test[:,:,c]

  # getting meta
  seqlen = np.size(X_test,1)
  d = np.size(X_test,2)
  num_classes = 8
  num_seq_test = np.size(X_test,0)
  del a, b, c

  ## DUMMY -> CONCAT ##
  vals = np.arange(0,8)
  labels_new = np.zeros((num_seq_test,seqlen))
  for i in range(np.size(labels_test,axis=0)):
    labels_new[i,:] = np.dot(labels_test[i,:,:], vals)
  labels_new = labels_new.astype('int32')
  labels_test = labels_new

  len_test = np.sum(mask_test, axis=1)

  if SAVE_DATASETS:
    save_raw_dataset(data=Y_test[:,:,np.arange(0,22)], Y=Y_test, masks=mask_test, targets=labels_test, seq_lengths=len_test, is_test=True)

  return X_test, mask_test, labels_test, num_seq_test, len_test

def save_raw_dataset(data, Y, masks, targets, is_test, seq_lengths=None, is_cullpdb=False):
  datadict = {0:1, 1:19, 2:3, 3:7, 4:12, 5:8, 6:5, 7:18, 8:15, 9:0, 10:6, 11:2, 12:17, 13:10, 14:11, 15:13, 16:14, 17:9, 18:4, 19:16, 20:1, 21:20}
  datargmax = np.argmax(data, axis=2)
  for i in range(len(data)):
    for j in range(len(data[0])):
      if datargmax[i][j] == 20:
        Y[i][j][np.arange(0,22)] = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
      datargmax[i][j] = datadict[datargmax[i][j]]
  if is_cullpdb:
    np.savez("data/SecPred/train_cullpdb_raw.npy", X_train=datargmax, t_train=targets, mask_train=masks)
    np.save("data/SecPred/train_cullpdb_no_x.npy", Y)
  else:
    if is_test:
      np.savez("data/SecPred/test_raw.npy", X_test=datargmax, t_test=targets, mask_test=masks, length_test=seq_lengths)
      np.save("data/SecPred/test_no_x.npy", Y)
    else:
      np.savez("data/SecPred/train_raw.npz", X_train=datargmax, t_train=targets, mask_train=masks)
      np.save("data/SecPred/train_no_x.npy", Y)
      

  


def get_casp(seq_len=None):
  X_casp, t_casp, mask_casp = casphandle.get_data()

  # getting meta
  seqlen = np.size(X_casp,1)
  d = np.size(X_casp,2)
  num_classes = 8

  ### ADDING BATCH PADDING ###
  num_add = 256 - X_casp.shape[0]
  X_add = np.zeros((num_add,seqlen,d))
  t_add = np.zeros((num_add,seqlen))
  mask_add = np.zeros((num_add,seqlen))
  #
  X_casp = np.concatenate((X_casp, X_add), axis=0).astype("float32")
  t_casp = np.concatenate((t_casp, t_add), axis=0).astype('int32')
  mask_casp = np.concatenate((mask_casp, mask_add), axis=0).astype("float32")
  if seq_len is not None:
    X_casp = X_casp[:, :seq_len]
    t_casp = t_casp[:, :seq_len]
    mask_casp = mask_casp[:, :seq_len]
  len_casp = np.sum(mask_casp, axis=1)
  len_casp[-num_add:] = np.ones((num_add,), dtype='int32')
  return X_casp, mask_casp, t_casp, len_casp


def load_data(is_cb513):
  if is_cb513:
    X_train, X_valid, t_train, t_valid, mask_train, \
    mask_valid, len_train, len_valid, num_seq_train = get_train_cb513()
    X_test, mask_test, t_test, num_seq_test, len_test = get_test()
  else:
    X_train, X_valid, X_test, t_train, t_valid, t_test, mask_train, \
    mask_valid, mask_test, len_train, len_valid, len_test, num_seq_train = get_train_cullpdb()

  X_casp, mask_casp, t_casp, len_casp = get_casp()

  dict_out = dict()
  dict_out['X_train'] = X_train
  dict_out['X_valid'] = X_valid
  dict_out['X_test'] = X_test
  dict_out['X_casp'] = X_casp
  dict_out['t_train'] = t_train
  dict_out['t_valid'] = t_valid
  dict_out['t_test'] = t_test
  dict_out['t_casp'] = t_casp
  dict_out['mask_train'] = mask_train
  dict_out['mask_valid'] = mask_valid
  dict_out['mask_test'] = mask_test
  dict_out['mask_casp'] = mask_casp
  dict_out['length_train'] = len_train
  dict_out['length_valid'] = len_valid
  dict_out['length_test'] = len_test
  dict_out['length_casp'] = len_casp
  return dict_out, num_seq_train

def chop_sequences(X, t, mask, length):
    max_len = np.max(length)
    return X[:, :max_len], t[:, :max_len], mask[:, :max_len]


class gen_data():
    def __init__(self, batch_size, is_cb513, data_fn=load_data):
        print("initializing data generator!")
        #self._num_iterations = num_iterations
        self._batch_size = batch_size
        self._data_dict, self._num_seq_train = load_data(is_cb513)
        self._seq_len = 700
        print(self._data_dict.keys())
        if 'X_train' in self._data_dict.keys():
            if 't_train' in self._data_dict.keys():
                print("Training is found!")
                self._idcs_train = list(range(self._data_dict['X_train'].shape[0]))
                self._num_features = self._data_dict['X_train'].shape[-1]
        if 'X_valid' in self._data_dict.keys():
            if 't_valid' in self._data_dict.keys():
                print("Valid is found!")
                self._idcs_valid = list(range(self._data_dict['X_valid'].shape[0]))
        if 'X_test' in self._data_dict.keys():
            if 't_test' in self._data_dict.keys():
                print("Test is found!")
                self._idcs_test = list(range(self._data_dict['X_test'].shape[0]))
        if 'X_casp' in self._data_dict.keys():
            if 't_casp' in self._data_dict.keys():
                print("CASP is found!")
                self._idcs_casp = list(range(self._data_dict['X_casp'].shape[0]))

    def _shuffle_train(self):
        np.random.shuffle(self._idcs_train)

    def _batch_init(self):
        batch_holder = dict()
        batch_holder["X"] = np.zeros((self._batch_size, self._seq_len, self._num_features), dtype="float32")
        batch_holder["t"] = np.zeros((self._batch_size, self._seq_len), dtype="int32")
        batch_holder["mask"] = np.zeros((self._batch_size, self._seq_len), dtype="float32")
        batch_holder["length"] = np.zeros((self._batch_size,), dtype="int32")
        return batch_holder

    def _chop_batch(self, batch, i=None):
        X, t, mask = chop_sequences(batch['X'], batch['t'], batch['mask'], batch['length'])
        if i is None:
            batch['X'] = X
            batch['t'] = t
            batch['mask'] = mask
        else:
            batch['X'] = X[:i]
            batch['t'] = t[:i]
            batch['mask'] = mask[:i]
        return batch

    def gen_valid(self):
        batch = self._batch_init()
        i = 0
        for idx in self._idcs_valid:
            batch['X'][i] = self._data_dict['X_valid'][idx]
            batch['t'][i] = self._data_dict['t_valid'][idx]
            batch['mask'][i] = self._data_dict['mask_valid'][idx]
            batch['length'][i] = self._data_dict['length_valid'][idx]
            i += 1
            if i >= self._batch_size:
                yield self._chop_batch(batch, i)
                batch = self._batch_init()
                i = 0
        if i != 0:
            yield self._chop_batch(batch, i)

    def gen_test(self):
        batch = self._batch_init()
        i = 0
        for idx in self._idcs_test[:512]:
            batch['X'][i] = self._data_dict['X_test'][idx]
            batch['t'][i] = self._data_dict['t_test'][idx]
            batch['mask'][i] = self._data_dict['mask_test'][idx]
            batch['length'][i] = self._data_dict['length_test'][idx]
            i += 1
            if i >= self._batch_size:
                yield self._chop_batch(batch, i)
                batch = self._batch_init()
                i = 0
        if i != 0:
            print(i)
            print(self._chop_batch(batch, i)['X'].shape)
            yield self._chop_batch(batch, i)

    def gen_casp(self):
        batch = self._batch_init()
        i = 0
        for idx in self._idcs_casp:
            batch['X'][i] = self._data_dict['X_casp'][idx]
            batch['t'][i] = self._data_dict['t_casp'][idx]
            batch['mask'][i] = self._data_dict['mask_casp'][idx]
            batch['length'][i] = self._data_dict['length_casp'][idx]
            i += 1
            if i >= self._batch_size:
                yield self._chop_batch(batch, i), i
                batch = self._batch_init()
                i = 0
        if i != 0:
            print(i)
            print(self._chop_batch(batch, i)['X'].shape)
            yield self._chop_batch(batch, i), i

    def gen_train(self):
        batch = self._batch_init()
        iteration = 0
        i = 0
        while True:
            # shuffling all batches
            self._shuffle_train()
            for idx in self._idcs_train:
                batch['X'][i] = self._data_dict['X_train'][idx]
                batch['t'][i] = self._data_dict['t_train'][idx]
                batch['mask'][i] = self._data_dict['mask_train'][idx]
                batch['length'][i] = self._data_dict['length_train'][idx]
                i += 1
                if i >= self._batch_size:
                    yield self._chop_batch(batch)
                    batch = self._batch_init()
                    i = 0
                    iteration += 1
#                    if iteration >= self._num_iterations:
#                        break
            else:
                continue
#            break
    
    def get_valid_data(self):
      X, t, mask = chop_sequences(self._data_dict['X_valid'], self._data_dict['t_valid'], self._data_dict['mask_valid'], self._data_dict['length_valid'].astype(dtype="int32"))
      return X, t, mask, self._data_dict['length_valid']

    def get_test_data(self):
      X, t, mask = chop_sequences(self._data_dict['X_test'], self._data_dict['t_test'], self._data_dict['mask_test'], self._data_dict['length_test'].astype(dtype="int32"))
      return X, t, mask, self._data_dict['length_test']
    


      
