import torch.utils.data as data
from params import args
import numpy as np
import torch as t

class EEGDataset(data.Dataset):
    def __init__(self, raw):
        self.raw = raw

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        return self.raw[index]