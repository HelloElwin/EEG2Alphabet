import torch.utils.data as data
import scipy.io as scio
from params import args
from utils import log
import numpy as np
import torch as t
import pickle

class EEGDataset(data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.transposed_label = np.transpose(self.label)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = t.tensor(self.data[index], dtype=t.float32)
        labl = t.tensor(self.label[index], dtype=t.float32)
        data = data + t.randn(data.shape) * 0.1 * data.mean() # add noise to prevent overfitting
        return data, labl

    def prepare_ssl_data(self):
        """
        Generate a list of index lists.
        The i-th list in self.ssl_data is indexes of the i-th letter.
        """
        self.ssl_data = []
        for i in range(26):
            idx = np.argwhere(self.transposed_label[i] == 1)
            self.ssl_data.append(idx)

    def get_ssl_data(self, label_idx):
        indexes = self.ssl_data[label_idx]
        indexes = np.random.permutation(indexes)[:128]
        return t.tensor(self.data[indexes], dtype=t.float32)

def slice_time(trn_data, tst_data, start_time=50, end_time=700):
        trn_data = trn_data[:,start_time:end_time,:]
        tst_data = tst_data[:,start_time:end_time,:]
        args.len_time = end_time - start_time
        return trn_data, tst_data

def split_data(raw, label, train_ratio=0.9):
    num_train = int(train_ratio * 300)

    trn_idx = np.array([])
    tst_idx = np.array([])

    for i in range(26):
        permu_idx = np.random.permutation(range(i * 300, (i + 1) * 300))
        trn_idx = np.append(trn_idx, permu_idx[:num_train], axis=0)
        tst_idx = np.append(tst_idx, permu_idx[num_train:], axis=0)

    trn_data = raw[trn_idx.astype('int32')]
    tst_data = raw[tst_idx.astype('int32')]
    trn_label = label[trn_idx.astype('int32')]
    tst_label = label[tst_idx.astype('int32')]

    return trn_data, trn_label, tst_data, tst_label

def get_datasets():
    log('Loading and splitting data from raw...')
    raw_info = scio.loadmat('./dataset/' + args.data + '.mat')
    raw = raw_info["data"] # (24, 801, 7800)
    raw = np.swapaxes(raw, 0, -1)
    label = np.squeeze(raw_info["label"])
    newLabel = np.zeros((7800, 26))
    newLabel[list(range(7800)), label - 1] = 1
    label = newLabel

    trn_data, trn_label, tst_data, tst_label = split_data(raw, label)
    pickle.dump((trn_data, tst_data, trn_label, tst_label), open('./eeg_dataset.pkl', 'wb'))
    trn_data, tst_data = slice_time(trn_data, tst_data)

    return trn_data, trn_label, tst_data, tst_label

