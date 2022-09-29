# EEG Alphabet Classifier

## Train Model

Train the EEGNet version, highest precision is about 14%.

`python labcode_eegnet.py --dropout 0.3`

Train the Transformer & Self-Supervised Learning version, highest precision is about 8.9%.

`python labcode_ssl.py --reg 1e-6 --reg_cont 1e-8`

## Methods

1. `labcode_ssl.py` 两个 Transformer 建模，拉近同一样本的两个表征
2. `labcode_eegnet.py` 使用 EEGNet
3. `labcode_clustering.py` 建模方法同上，拉近同一标签的样本，推远不同标签的样本# EEG Alphabet Classifier

## Train Model

Train the EEGNet version, highest precision is about 14%.

`python labcode_eegnet.py --dropout 0.3`

Train the Transformer & Self-Supervised Learning version, highest precision is about 8.9%.

`python labcode_ssl.py --reg 1e-6 --reg_cont 1e-8`

### Preprocessing

We use [filter-bank](https://github.com/HelloElwin/EEG/tree/main/preprocessing):

| Frequency | Link                                                         |
| --------- | ------------------------------------------------------------ |
| 2-4 Hz    | https://connecthkuhk-my.sharepoint.com/:u:/g/personal/nebula_connect_hku_hk/Ec9GC54t8pBEoGBzsDfoqLABZUPHyWpzyPUm1ZjYu5BLUQ?e=wKV1Br |
| 4-8 Hz    | https://connecthkuhk-my.sharepoint.com/:u:/g/personal/nebula_connect_hku_hk/EXFzzPKUFndEgDzWabPcFdUBA1B55v11hd2bkbcrpGhQGQ?e=U8zFcF |
| 8-12 Hz   | https://connecthkuhk-my.sharepoint.com/:u:/g/personal/nebula_connect_hku_hk/EVsLUCk_chZOlMm694SAyHEBHDzxQ0nlHi_UCK4llGwxwg?e=4uRljB |
| 12- 15 Hz | https://connecthkuhk-my.sharepoint.com/:u:/g/personal/nebula_connect_hku_hk/EWEnpqbNMxVBnth72JKjs5EBIYy49A-xKYYaPun_d4B_Kg?e=BnQxQC |



## Methods

1. `labcode_ssl.py` 两个 Transformer 建模，拉近同一样本的两个表征
2. `labcode_eegnet.py` 使用 EEGNet
3. `labcode_clustering.py` 建模方法同上，拉近同一标签的样本，推远不同标签的样本


## Acknowledgement

The code for EEGNet is based on [[this repo]](https://github.com/vlawhern/arl-eegmodels).

## Acknowledgement

The code for EEGNet is based on [[this repo]](https://github.com/vlawhern/arl-eegmodels).

