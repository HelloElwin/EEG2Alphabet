#encoding:utf-8
from data import EEGDataset, get_datasets
import torch.utils.data as dataloader
from setproctitle import setproctitle
from params import args
from EEGNet import *
from model import *
from utils import *
import numpy as np
import torch as t

setproctitle('eeg@elwin')

t.manual_seed(19260817)
np.random.seed(19260817)

trn_data, trn_label, tst_data, tst_label = get_datasets()

trn_data = np.swapaxes(trn_data, -1, -2)
tst_data = np.swapaxes(tst_data, -1, -2)

trn_data = np.expand_dims(trn_data, axis=-1)
tst_data = np.expand_dims(tst_data, axis=-1)

print(trn_data.shape)

trn_data *= 1000
tst_data *= 1000

model = EEGNet(nb_classes = 26, Chans = args.num_chan, Samples = args.len_time,
               dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16,
               dropoutType = 'Dropout')

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics = ['accuracy'])

fittedModel = model.fit(trn_data, trn_label, batch_size = 16, epochs = 100, 
                        verbose = 2, validation_data=(tst_data, tst_label))
                        # callbacks=[checkpointer], class_weight=class_weights)

probs = model.predict(tst_data)
preds = probs.argmax(axis = -1)
acc   = np.mean(preds == tst_label.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))
