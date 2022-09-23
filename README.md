# EEG Alphabet Classifier

Run the ssl version, highest precision is 8.4%.

`python labcode_ssl.py --reg 1e-6 --reg_cont 1e-8`

## Method

1. `labcode_ssl.py` 两个 Transformer 建模，拉近同一样本的两个表征
2. `labcode_clustering.py` 建模方法同上，拉近同一标签的样本，推远不同标签的样本`
