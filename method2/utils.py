import sys
import datetime
import numpy as np
from params import args

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
