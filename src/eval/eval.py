from eval_2d import run_on_exp as run_on_exp_2d
from eval_3d import run_on_exp as run_on_exp_3d
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_name', type=str, help="Name of experiment")
	parser.add_argument('--gpu', type=int, default=0, help="ID of CUDA GPU to use")

	parser.add_argument('--unseen', action = 'store_true')

	args = parser.parse_args()
	run_on_exp_3d(args.exp_name, gpu=args.gpu)
	run_on_exp_2d(args.exp_name, gpu=args.gpu)
