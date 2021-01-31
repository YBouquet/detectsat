import argparse
import os

parser = argparse.ArgumentParser(description='Satellite Streaks Detection')

parser.add_argument('--i', type=str)
parser.add_argument('--h', type=str)
parser.add_argument('--o', type = str)
parser.add_argument('--n', type = int, default = 1)
parser.add_argument('--subcrop_i', type= int, default = 0)
parser.add_argument('--subcrop_j', type= int, default = 0)
parser.add_argument('--hough', type = int, default = 200)
parser.add_argument('--load_lines', dest='load_lines', action='store_true')
parser.add_argument('--no-save_lines', dest='save_lines', action='store_false')
parser.set_defaults(load_lines=False)
parser.set_defaults(save_lines=True)

#parser.add_argument('--learning_rate', type = float, default = 5e-4, help = 'Learning rate for the optimizer')
args = parser.parse_args()

def get_args():
    return args
