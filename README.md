# EEG Alphabet Classifier

In this repository, two solutions are proposed and implemented for the *HK EEG + AI competition*.

## Usage

### Requirements

```markdown
python==3.8
pytorch>=1.12.1
numpy>=1.21.5
scipy>=1.7.3
```

### Evaluation

Firstly, please put the evaluation dataset in the folder `dataset` and rename it as `data.mat`. The whole dataset will be used for evaluation.

Then please run `python src/main.py --eval` to evaluate the model.

## Method

We build two transformers to encode information from both time (temporal) and channel (spatial) domain of the EEG data. Embeddings from these two views are concatenated and fed into an MLP for the final classification task. To tackle the noise problem of EEG data and introduce more supervision signal, we further consider the temporal and spatial embeddings as two different views and perform contrastive learing between them.

### Train the Model

`python src/main.py --dropout 0.3`

## Acknowledgement

The code for EEGNet is based on [[this repo]](https://github.com/vlawhern/arl-eegmodels).
