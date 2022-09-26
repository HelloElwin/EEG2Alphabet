import torch.utils.data as data
import scipy.io as scio
from params import args
from utils import log
import numpy as np
import torch as t
import pickle

file_path = './data_EEG_AI.mat'

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
        data = data + t.randn(data.shape) * 0.01 # add noise to prevent overfitting
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


def get_datasets():
    """
    Get the dataset and split it into train and test set by 9:1.
    """
    try:
        trn_data, tst_data, trn_label, tst_label = pickle.load(open('./eeg_dataset.pkl', 'rb'))
        log('Loading data from pkl...')
        end_time = 600
        start_time = 200
        trn_data = trn_data[:,start_time: end_time,:]
        tst_data = tst_data[:,start_time: end_time,:]
        trn_label = trn_label[:,start_time: end_time,:]
        tst_label = tst_label[:,start_time: end_time,:]
        args.len_time = end_time - start_time
    except:
        log('Loading and splitting data from raw...')
        raw_info = scio.loadmat(file_path)
        raw = raw_info["data"] # (24, 801, 7800)
        raw = np.swapaxes(raw, 0, -1)
        label = np.squeeze(raw_info["label"])
        newLabel = np.zeros((7800, 26))
        newLabel[list(range(7800)), label - 1] = 1
        label = newLabel

        trn_idx = np.array([])
        tst_idx = np.array([])

        for i in range(26):
            permu_idx = np.random.permutation(range(i * 300, (i + 1) * 300))
            trn_idx = np.append(trn_idx, permu_idx[permu_idx.shape[0] // 10:], axis=0)
            tst_idx = np.append(tst_idx, permu_idx[:permu_idx.shape[0] // 10], axis=0)

        trn_data = raw[trn_idx.astype('int32')]
        tst_data = raw[tst_idx.astype('int32')]
        trn_label = label[trn_idx.astype('int32')]
        tst_label = label[tst_idx.astype('int32')]
        '''
        permutation_idx = np.random.permutation(len(label))
        trn_data = raw[permutation_idx][len(label) // 10:]
        tst_data = raw[permutation_idx][:len(label) // 10]
        trn_label = label[permutation_idx][len(label) // 10:]
        tst_label = label[permutation_idx][:len(label) // 10]
        '''

        pickle.dump((trn_data, tst_data, trn_label, tst_label), open('./eeg_dataset.pkl', 'wb'))

    return EEGDataset(trn_data, trn_label), EEGDataset(tst_data, tst_label)
