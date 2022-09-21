import sys
import datetime
import torch as t
from params import args
import torch.nn.functional as F

def calc_reg_loss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	return ret

def calc_contrastive_loss(emb1, emb2):
    pos = t.exp(t.sum(emb1 * emb2, axis=1))
    neg = t.reduce_sum(t.exp(emb1 @ t.transpose(emb2)), axis=1)
    scr = t.reduce_sum(-t.log(pos / (neg + 1e-8) + 1e-8))
    return scr

def log(info, online=False):
	time = datetime.datetime.now()
	info = f'{time} {info}'
	if online: print(info, end='\r')
	else: print(info)
	sys.stdout.flush()
