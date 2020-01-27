import torch
import numpy as np

def make_one_hot(batched_labels):
        shape = batched_labels.shape
        print(shape)
        one_hot = torch.zeros(shape[0], shape[1], 4)
        print(one_hot.shape)
        for row_idx in range(shape[0]):
          one_hot[row_idx, np.arange(shape[1]), batched_labels[row_idx]] = 1
        return one_hot

a = torch.tensor([[1,2,3],[3,2,1]])
print( make_one_hot(a) )