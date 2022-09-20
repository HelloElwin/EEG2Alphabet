import torch.utils.data as data
from params import args
import numpy as np
import torch as t
import scipy.io as scio

file_path = './data_EEG_AI.mat'

class EEGDataset():
    def __init__(self):
        self.raw_info = scio.loadmat(file_path)
        self.raw = self.raw_info["data"] # (24, 801, 7800)
        self.raw = np.swapaxes(self.raw, 0, -1)
        self.label = np.squeeze(self.raw_info["label"])
        self.newLabel = np.zeros((7800, 26))
        self.newLabel[list(range(7800)), self.label - 1] = 1
        # for i in range(200, 400):
            # print(self.newLabel[i])
        self.label = self.newLabel
            

    def __len__(self):
        return self.raw.shape[0]

    def __getitem__(self, index):
        return (t.tensor(self.raw[index], dtype=t.float32), t.tensor(self.label[index], dtype=t.float32))

def split_dataset():
    pass

a = EEGDataset()
