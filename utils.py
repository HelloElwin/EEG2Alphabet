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

def log(info, online=False):
	time = datetime.datetime.now()
	info = f'{time} {info}'
	if online: print(info, end='\r')
	else: print(info)
	sys.stdout.flush()
