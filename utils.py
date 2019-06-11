import numpy as np
import random
import torch
from collections import OrderedDict
from torch.autograd import Variable

def iterate_minibatches(inputs, targets, masks, targets_mem, unk_mem, batchsize, shuffle=True, sort_len=True, sample_last_batch=True):
  """ Generate minibatches of a specific size 
  Arguments:
    inputs -- numpy array of the encoded protein data. Shape: (n_samples, seq_len, n_features)
    targets -- numpy array of the targets. Shape: (n_samples,)
    masks -- numpy array of the protein masks. Shape: (n_samples, seq_len)
    batchsize -- integer, number of samples in each minibatch.
    shuffle -- boolean, shuffle the samples in the minibatches. (default=False)
    sort_len -- boolean, sort the minibatches by sequence length (faster computation, just for training). (default=True) 
  Outputs:
  list of minibatches for protein sequences, targets and masks.

  """ 
  assert len(inputs) == len(targets)

  # Calculate the sequence length of each sample
  len_seq = np.apply_along_axis(np.bincount, 1, masks.astype(np.int32))[:,-1]

  # Sort the sequences by length
  if sort_len:
    indices = np.argsort(len_seq) #[::-1][:len(inputs)] #sort and reverse to get in decreasing order
  else:
    indices = np.arange(len(inputs))

  # Generate minibatches list
  f_idx = len(inputs) % batchsize
  idx_list = list(range(0, len(inputs) - batchsize + 1, batchsize))
  last_idx = None
  if f_idx != 0 and sample_last_batch:
    last_idx = idx_list[-1] + batchsize
    idx_list.append(last_idx)

  # Shuffle the minibatches
  if shuffle:
    random.shuffle(idx_list)

  # Split the data in minibatches
  for start_idx in idx_list:
    if start_idx == last_idx:
      rand_samp = batchsize - f_idx
      B = np.random.randint(len(inputs),size=rand_samp)
      excerpt = np.concatenate((indices[start_idx:start_idx + batchsize], B))
    else:
      excerpt = indices[start_idx:start_idx + batchsize]
    max_prot = np.amax(len_seq[excerpt])

    # Crop batch to maximum sequence length
    if sort_len:
      in_seq = inputs[excerpt][:,:max_prot]
      in_mask = masks[excerpt][:,:max_prot]
    else:
      in_seq = inputs[excerpt][:,:max_prot]
      in_mask = masks[excerpt][:,:max_prot]

    in_target = targets[excerpt]
    in_target_mem = targets_mem[excerpt]
    in_unk_mem = unk_mem[excerpt]
    shuf_ind = np.arange(batchsize)

    # Return a minibatch of each array
    yield in_seq[shuf_ind], in_target[shuf_ind], in_mask[shuf_ind], in_target_mem[shuf_ind], in_unk_mem[shuf_ind]

# Used for attention mechanism
def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or int(length.max().item())
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    mask = mask.float()
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    
    return mask

def length_to_negative_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or int(length.max().item())
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    mask = mask.float()
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
        
    mask = (mask - 1) * 10e6
    return mask

class ResultsContainer():
  """
    A simple class containing all type of results 
    from main script to be used in notebooks for vizualisation
  """
  def __init__(self):
    # util
    self.best_val_acc = 0

    # epochs
    self.epochs = 0
    self.loss_training = []
    self.loss_validation = []
    self.acc_training = []
    self.acc_validation = []

    # final
    self.alphas = None
    self.seq_lengths = None
    self.targets = None
    self.cf_test = None
    self.cf_mem_test = None
    self.test_acc = 0
    self.test_mem_acc = 0

  def append_epoch(self, train_loss, val_loss, train_acc, val_acc):
    self.loss_training.append(train_loss)
    self.loss_validation.append(val_loss)
    self.acc_training.append(train_acc)
    self.acc_validation.append(val_acc)
    self.epochs += 1

  def set_final(self, alph, seq_len, targets, cf, cf_mem, acc, acc_mem):
    self.alphas = alph
    self.seq_lengths = seq_len
    self.targets = targets
    self.cf_test = cf
    self.cf_mem_test = cf_mem
    self.test_acc = acc
    self.test_mem_acc = acc_mem

def tensor_to_onehot(y, n_dims=20):
  y_tensor = y.data if isinstance(y, Variable) else y
  y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
  n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
  y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
  y_one_hot = y_one_hot.view(*y.shape, -1)
  return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

# https://github.com/rdipietro/pytorch/blob/4c00324affb8c6d53d4362e321ea0e99ede6cfde/torch/nn/utils/rnn.py
def reverse_padded_sequence(inputs, lengths, batch_first=False):
    """Reverses sequences according to their lengths.
    
    `inputs` should have size `T x B x *` if `batch_first` is `False`, or
    `B x T x *` if `True`. `T` is the length of the longest sequence (or
    larger), `B` is the batch size, and `*` is any number of dimensions
    (including 0).
    
    Arguments:
        inputs (Tensor): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if `True`, `inputs` should be `B x T x *`.
        
    Returns:
        A Tensor with the same size as `inputs`, but with each sequence
        reversed according to its length.
    """
    if batch_first:
        inputs = inputs.transpose(0, 1)
    max_length, batch_size = inputs.size(0), inputs.size(1)
    if len(lengths) != batch_size:
        raise ValueError('inputs is incompatible with lengths.')
    ind = [list(reversed(range(0, length))) + list(range(length, max_length))
           for length in lengths]
    ind = torch.LongTensor(ind).transpose(0, 1)
    for dim in range(2, inputs.dim()):
        ind = ind.unsqueeze(dim)
    ind = ind.expand_as(inputs)
    if inputs.is_cuda:
        ind = ind.cuda(inputs.get_device())
    reversed_inputs = torch.gather(inputs, 0, ind)
    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs



# https://gist.github.com/the-bass/0bf8aaa302f9ba0d26798b11e4dd73e3
def rename_state_dict_keys(state_dict, key_transformation):
    """
    state_dict         -> State dict object.
    key_transformation -> Function that accepts the old key names of the state
                          dict as the only argument and returns the new key name.
    Example:
    Rename the key `layer.0.weight` `layer.1.weight` and keep the names of all
    other keys.
    ```py
    def key_transformation(old_key):
        if old_key == "layer.0.weight":
            return "layer.1.weight"
        return old_key
    rename_state_dict_keys(state_dict, key_transformation)
    ```
    """
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value

    return new_state_dict

def do_layer_norm(tensor, mask):
    # Prepare variables
    broadcast_mask = mask.unsqueeze(-1)
    input_dim = tensor.size(-1)
    num_elements_not_masked = torch.sum(mask) * input_dim

    # Do normalization
    tensor_masked = tensor * broadcast_mask
    mean = torch.sum(tensor_masked) / num_elements_not_masked
    variance = torch.sum(((tensor_masked - mean) * broadcast_mask)**2) / num_elements_not_masked
    return (tensor - mean) / torch.sqrt(variance + 1E-12)

if __name__ == "__main__":
  batch_size = 7
  seq_len = 11
  classes = 4

  y = torch.LongTensor(batch_size,seq_len).random_(classes)

  one_hot = tensor_to_onehot(y,classes)

  for i in range(y.shape[0]):
    for j in range(y.shape[1]):
      assert(one_hot[i,j,y[i,j]] == 1.0)
