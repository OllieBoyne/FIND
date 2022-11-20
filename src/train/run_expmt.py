"""Run experiment from config file."""
import init_paths
import os, yaml
from src.train.opts import Opts
from src.train.train import run
from collections import defaultdict
import multiprocessing as mp
from multiprocessing import Process


class Exp:
	def __init__(self, opts):
		self.opts = opts

	def set_gpu(self, gpu=0):
		self.opts.set_option('gpu', gpu)

	def run_silent(self):
		self.opts.set_option('silent', True)

	def run(self):
		run(self.opts)


class Queue:
	def __init__(self, expmt_list):
		self.expmt_list = expmt_list

	def run(self):
		for expmt in self.expmt_list:
			expmt.run()


def main(cfg_file):
	with open(cfg_file) as infile:
		cfg = yaml.load(infile, Loader=yaml.FullLoader)

	name = cfg['EXPERIMENT_NAME']
	out_dir = os.path.join('exp', name)
	os.makedirs(out_dir, exist_ok=True)

	opts = Opts()
	opts.set_option('save_dir', out_dir)

	# Set all common arguments
	for setting, value in cfg['COMMON_ARGS'].items():
		opts.set_option(setting, value)

	load_from_reg = cfg['REG_FIRST_ONLY']
	load_from_reg_loc = None  # If copying reg between models, load from this location

	experiments = []

	# Run each experiment
	for expmt_name in cfg['EXPERIMENTS']:
		expmt_opts = opts.copy_opts()
		opt_dict = cfg['EXPERIMENTS'][expmt_name]
		for setting, value in opt_dict.items():
			expmt_opts.set_option(setting, value)

		# Load reg from a previous experiment if required
		if load_from_reg:
			if load_from_reg_loc and not opt_dict.get('reg',
													  False):  # If this experiment doesn't have a reg stage, and there is a reg to load from
				expmt_opts.set_option('load_model', load_from_reg_loc)  # Load this reg
			else:  # If no reg made yet, set this model to produce the reg
				assert opt_dict[
					'reg'], "load_from_reg setting used, but this requires first experiment to have --reg flag on!"
				load_from_reg_loc = os.path.join(out_dir, expmt_name, 'reg.pth')

		expmt_opts.set_option('model_name', expmt_name)  # Set save location
		expmt_opts.validate_training()  # Check the training settings are all valid

		experiments.append(Exp(expmt_opts))  # Add to queue

	# CREATE QUEUE
	if not cfg['USE_THREADING']:
		Queue(experiments).run()  # Run all experiments sequentially

	else:  # use threading
		gpus = cfg['GPUS']
		ngpus = len(gpus)  # A thread for each CUDA GPU
		if load_from_reg:  # If registration is from first experiment, need to run this one first before queues
			expmt_0 = experiments[0]
			expmt_0.set_gpu(gpus[0])
			expmt_0.run()
			experiments = experiments[1:]  # remove first experiment from queue

		queues = defaultdict(list)
		for n, expmt in enumerate(experiments):
			gpu = gpus[n % ngpus]
			expmt.set_gpu(gpu)  # configure experiment to run on this gpu
			expmt.run_silent()  # As there will be multiple threads, log directly to log.txt, not to stdout
			queues[gpu].append(expmt)  # add this experiment to that gpu's queue

		# Start thread running for each GPU
		for gpu in gpus:
			thread = Process(target=Queue(queues[gpu]).run)
			thread.start()


if __name__ == '__main__':
	mp.set_start_method('spawn')
	opts = Opts()
	if opts.exp_name is None:
		raise ValueError("This script requires command line arg exp_name")

	cfg_file = f"cfgs/{opts.exp_name}.yaml"
	assert os.path.isfile(cfg_file), f"No experiment found: `{opts.exp_name}`"
	main(cfg_file)
