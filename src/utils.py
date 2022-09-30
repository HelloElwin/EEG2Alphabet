import sys
import datetime
import torch as t
import numpy as np
from params import args
import torch.nn.functional as F

def calc_reg_loss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	return ret

def calc_contrastive_loss(emb1, emb2):
    emb1 = F.normalize(emb1)
    emb2 = F.normalize(emb2)
    score = -t.log(t.sum(t.exp(t.sum(emb1 * emb2, axis=1))) + 1e-8)
    # neg = t.sum(t.exp(emb1 @ emb2.T), axis=1)
    # scr = t.sum(-t.log(pos / (neg + 1e-8) + 1e-8))
    return score 

def log(info, online=False, bold=False):
    time = datetime.datetime.now()
    info = f'{time} {info}'
    if bold:
        info = '\033[1m' + info + '\033[0m'
    if online:
        print(info, end='\r')
    else:
        print(info)
    sys.stdout.flush()

def get_pos_emb(seq_len, dim):

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (dim_i // 2) / dim) for dim_i in range(dim)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(seq_len)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return t.FloatTensor(sinusoid_table).unsqueeze(0)
