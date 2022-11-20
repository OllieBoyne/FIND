import init_paths
import torch
import yaml

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('src/cfg.yaml') as infile:
	cfg = yaml.load(infile, Loader=yaml.FullLoader)