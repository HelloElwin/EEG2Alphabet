# EEG Alphabet Classifier

In this repository, two solutions are proposed and implemented for the *HK EEG + AI competition*.

## Usage

### Requirements

Environment for Method 1:

```markdown
python==3.8
pytorch>=1.12.1
numpy>=1.21.5
scipy>=1.7.3
```

Environment for Method 2:

```markdown
python==3.8
tensorflow>=2.10.0
numpy>=1.21.5
scipy>=1.7.3
```

### Evaluation

Please put the evaluation dataset in the folder `dataset` and rename it as `data.mat`. The whole dataset will be used for evaluation.

- For method 1 please run `python ./method1/main.py --eval` 

- For method 2 please run `python ./method2/main.py --eval 2> .error.log`

## Methods

### 1 Spatial-Temporal Transformer with Contrastive Learning

Link to code: [[Method 1]](./method1/)

We build two transformers to encode information from both time (temporal) and channel (spatial) domain of the EEG data. Embeddings from these two views are concatenated and fed into an MLP for the final classification task. To tackle the noise problem of EEG data and introduce more supervision signal, we further consider the temporal and spatial embeddings as two contrastive views and perform contrastive learing between them. Additionally, to further reduce noise in the data, we only choose the `0~2600ms` part of every sample.

### 2 EEGNet-based Classification

Link to code: [[Method 2]](./method2/)

We also build an classifier based on EEGNet, the state-of-the-art model for EEG classification task. Experiments on model structure and hyperparameter are performed to better apply the model to our dataset. Additionally, to further reduce noise in the data, we only choose the `0~2600ms` part of every sample. 

## Train the Models

Method 1:

`python method1/main.py --dropout 0.3`

Method 2:

`python method2/main.py --reg 1e-6 --reg_cont 1e-8 --len_time 650` 

### Data Preprocessing

We use [filter-bank](https://github.com/HelloElwin/EEG/tree/main/preprocessing):

| Frequency | Link                                                         |
| --------- | ------------------------------------------------------------ |
| 2-4 Hz    | https://connecthkuhk-my.sharepoint.com/:u:/g/personal/nebula_connect_hku_hk/Ec9GC54t8pBEoGBzsDfoqLABZUPHyWpzyPUm1ZjYu5BLUQ?e=wKV1Br |
| 4-8 Hz    | https://connecthkuhk-my.sharepoint.com/:u:/g/personal/nebula_connect_hku_hk/EXFzzPKUFndEgDzWabPcFdUBA1B55v11hd2bkbcrpGhQGQ?e=U8zFcF |
| 8-12 Hz   | https://connecthkuhk-my.sharepoint.com/:u:/g/personal/nebula_connect_hku_hk/EVsLUCk_chZOlMm694SAyHEBHDzxQ0nlHi_UCK4llGwxwg?e=4uRljB |
| 12- 15 Hz | https://connecthkuhk-my.sharepoint.com/:u:/g/personal/nebula_connect_hku_hk/EWEnpqbNMxVBnth72JKjs5EBIYy49A-xKYYaPun_d4B_Kg?e=BnQxQC |
| 0.1-20 Hz | https://connecthkuhk-my.sharepoint.com/:u:/g/personal/nebula_connect_hku_hk/EddN6QuO_jRBuEQu-5YnNAsBbvC1HI-DnC2-JwYLu_EQPg?e=V4oszj |


## Acknowledgement

The code for EEGNet is based on [[this repo]](https://github.com/vlawhern/arl-eegmodels).
