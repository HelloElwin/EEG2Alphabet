import torch.utils.data as data
from params import args
import numpy as np
import torch as t
import scipy.io as scio

file_path = './data_EEG_AI.mat'

class EEGDataset(data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = t.tensor(self.data[index], dtype=t.float32)
        labl = t.tensor(self.label[index], dtype=t.float32)
        data = data + t.randn(data.shape) * 0.01 # add noise to prevent overfitting
        return data, labl

def get_datasets():
    """
    Get the dataset and split it into train and test set by 9:1
    """
    raw_info = scio.loadmat(file_path)
    raw = raw_info["data"] # (24, 801, 7800)
    raw = np.swapaxes(raw, 0, -1)
    label = np.squeeze(raw_info["label"])
    newLabel = np.zeros((7800, 26))
    newLabel[list(range(7800)), label - 1] = 1
    label = newLabel
    
    permutation_idx = np.random.permutation(len(label))
    trn_data = raw[permutation_idx][len(label) // 10:]
    tst_data = raw[permutation_idx][:len(label) // 10]
    trn_label = label[permutation_idx][len(label) // 10:]
    tst_label = label[permutation_idx][:len(label) // 10]

    return EEGDataset(trn_data, trn_label), EEGDataset(tst_data, tst_label)
