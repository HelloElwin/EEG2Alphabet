import torch.utils.data as data
from params import args
import numpy as np
import torch as t
import scipy.io as scio

File_path = './data_EEG_AI.mat'

class EEGDataset():
    def __init__(self):
        self.raw_info = scio.loadmat(File_path)
        self.raw = self.raw_info["data"] # (24, 801, 7800)
        self.raw = np.swapaxes(self.raw, 0, -1)
        self.label = np.squeeze(self.raw_info["label"])

    def __len__(self):
        return len(self.raw.shape[0])

    def __getitem__(self, index):
        return (t.tensor(self.raw[index]), t.tensor(self.label[index]))

a = EEGDataset()
