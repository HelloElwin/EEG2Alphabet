# EEG Alphabet Classifier

## Usage

### Requirements

```markdown
python==3.8
tensorflow>=2.10.0
numpy>=1.21.5
scipy>=1.7.3
```

### Evaluation

Please put the evaluation dataset in the folder `dataset` and rename it as `data.mat`. The whole dataset will be used for evaluation.

Please run:

`python src/main.py --eval 2> .error.log`

## Method

We build an classifier based on EEGNet, the state-of-art model for EEG classification task. Experiments on model structure and hyperparameter are performed to better apply the model to our dataset.

### Train the Model

`python src/main.py --reg 1e-6 --reg_cont 1e-8 --len_time 650` 

## Acknowledgement

The code for EEGNet is based on [[this repo]](https://github.com/vlawhern/arl-eegmodels).
