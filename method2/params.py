import argparse

def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Parameters')
    parser.add_argument('--data', default='data', type=str, help='name of dataset')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--len_time',  default=650, type=int, help='number of time ')
    parser.add_argument('--num_chan',  default=24, type=int, help='number of channels')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
    return parser.parse_args()
args = ParseArgs()
