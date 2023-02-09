import init_paths
import torch
import yaml
import os

def up1(p):
	return os.path.split(p)[0]

FIND_dir = up1(up1(os.path.dirname(os.path.realpath(__file__))))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(os.path.join(FIND_dir, 'src/cfg.yaml')) as infile:
	cfg = yaml.load(infile, Loader=yaml.FullLoader)