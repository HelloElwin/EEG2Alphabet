# EEG Alphabet Classifier

## Train Model

Train the EEGNet version, highest precision is about 14%.

`python labcode_eegnet.py --dropout 0.3`

Train the Transformer & Self-Supervised Learning version, highest precision is about 8.9%.

`python labcode_ssl.py --reg 1e-6 --reg_cont 1e-8`

## Methods

1. `labcode_ssl.py` 两个 Transformer 建模，拉近同一样本的两个表征
2. `labcode_eegnet.py` 使用 EEGNet
3. `labcode_clustering.py` 建模方法同上，拉近同一标签的样本，推远不同标签的样本

## Acknowledgement

The code for EEGNet is based on [[this repo]](https://github.com/vlawhern/arl-eegmodels).

