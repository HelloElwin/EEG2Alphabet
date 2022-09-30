#encoding:utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from params import args
from data import * 
from model import *
from utils import *
import numpy as np

np.random.seed(19260817)
tf.random.set_seed(19260817)

model = EEGNet(nb_classes = 26, Chans = args.num_chan, Samples = args.len_time,
        dropoutRate = args.dropout, kernLength = 32, F1 = 8, D = 2, F2 = 16,
        dropoutType = 'Dropout')

model.compile(loss='categorical_crossentropy', optimizer='adam',
        metrics = ['accuracy'])

if args.eval:
    tst_data, tst_label = get_eval_datasets()
    tst_data = np.swapaxes(tst_data, -1, -2)
    tst_data = np.expand_dims(tst_data, axis=-1)
    tst_data *= 1000
    model.load_weights('./ckpt/method2_best.ckpt')
    probs = model.predict(tst_data)
    preds = probs.argmax(axis = -1)
    acc   = np.mean(preds == tst_label.argmax(axis=-1))
    log(f"Classification accuracy = {acc}", bold=True)

else: 
    trn_data, trn_label, tst_data, tst_label = get_datasets()

    trn_data = np.swapaxes(trn_data, -1, -2)
    tst_data = np.swapaxes(tst_data, -1, -2)

    trn_data = np.expand_dims(trn_data, axis=-1)
    tst_data = np.expand_dims(tst_data, axis=-1)

    trn_data *= 1000
    tst_data *= 1000

    checkpointer = ModelCheckpoint(filepath='./best.ckpt', verbose=1, save_best_only=True)

    fittedModel = model.fit(trn_data, trn_label, batch_size = 16, epochs = 100, verbose = 2,
            validation_data=(tst_data, tst_label), callbacks=[checkpointer])

    model.load_weights('./ckpt/method2_best.ckpt')

    probs = model.predict(tst_data)
    preds = probs.argmax(axis = -1)
    acc   = np.mean(preds == tst_label.argmax(axis=-1))
    log(f"Classification accuracy = {acc}", bold=True)
