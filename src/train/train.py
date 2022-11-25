"""Train a FIND model."""
import init_paths
import sys, os
import trimesh

import torch.optim
from torch.utils.data import DataLoader

from src.data.dataset import Foot3DDataset, BatchCollator, NoTextureLoading
from src.model.model import Model, ModelWithLoss
from src.utils.utils import cfg
from src.train.opts import Opts
from src.train.trainer import Trainer


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # For debugging purposes

class Logger(object):
	def __init__(self, logfile, overwrite_log=False):
		self.terminal = sys.stdout
		self.logfile = logfile
		self.log = []
		if overwrite_log:
			open(logfile, 'w').close()
		else:
			with open(logfile) as infile:
				self.log = [l for l in infile.readlines()]

	def save(self):
		with open(self.logfile, "w", encoding="utf-8") as log:
			log.writelines(self.log)

	def write(self, message):
		self.terminal.write(message)
		self.log.append(message + '\n')

	def add_to_log(self, message):
		if message:
			self.log.append(message + '\n')

	def flush(self):
		# this flush method is needed for python 3 compatibility.
		# this handles the flush command by doing nothing.
		# you might want to specify some extra behavior here.
		pass





def train_network(args, model, trainer, logger, train_dataset, val_dataset, out_dir='', checkpoint_dir='',
				  num_epochs=None, save_every=None, val_every=None, plot_name='net', val_only=False):
	val_best = 1e3
	best_epoch = 0

	for epoch in range(num_epochs):
		this_epoch = lambda v: epoch % v == 0
		is_last = epoch == num_epochs - 1
		save_model = this_epoch(save_every) or is_last
		save_renders = save_model and not args.no_rendering

		model_kwargs = args.net_train_kwargs()
		# Render foot either if a rendering related loss, or saving model
		model_kwargs['render_foot'] = any(
			k in args.render_losses and v for k, v in model_kwargs.items()) or save_renders
		model_kwargs['save_renders'] = save_renders
		model_kwargs['render_dir'] = os.path.join(args.render_dir, 'train')
		os.makedirs(model_kwargs['render_dir'], exist_ok=True)
		model_kwargs['restyle_feature_maps'] = args.restyle_feature_maps
		model_kwargs['no_displacement'] = args.only_classifier_head
		model_kwargs['copy_mask_out'] = args.copy_over_masking
		model_kwargs['mask_out_pred_faces'] = args.mask_out_pred
		model_kwargs['gt_z_cutoff'] = args.gt_z_cutoff

		if not val_only:
			msg = trainer.train_epoch(epoch, save_model=save_model, model_kwargs=model_kwargs)
			logger.add_to_log(msg)

		if val_only or this_epoch(val_every) or is_last:
			model_kwargs['render_dir'] = os.path.join(args.render_dir, 'val')
			os.makedirs(model_kwargs['render_dir'], exist_ok=True)
			msg, res = trainer.val_epoch(epoch, model_kwargs=model_kwargs)
			logger.add_to_log(msg)

			if val_only and not (this_epoch(val_every) or is_last):
				continue  # Do not evaluate metrics every single epoch if val_only setting on

			metric = res.get(args.val_crit, 1e4)
			if metric <= val_best:
				best_epoch = epoch
				val_best = metric
				print(f"NEW BEST: epoch {epoch}, metric = {val_best:.4f}")
				trainer.best_epochs.append(epoch)

				model.save_model(out_dir=out_dir, fname='model_best')

				# Only export meshes on last epoch
				if is_last:
					trainer.export_meshes(export_loc=os.path.join(args.vis_dir,'train_meshes'), is_train=True, export_gt=False)
					trainer.export_meshes(export_loc=os.path.join(args.vis_dir,'val_meshes'), is_train=False, export_gt=False)

			else:
				print(f"NOT NEW BEST. epoch = {epoch}, metric = {metric:.4f}, best = {val_best:.4f}")

			trainer.plot(os.path.join(args.vis_dir, f'{plot_name}_train.png'))
			logger.save()

		if save_model:
			model.save_model(out_dir=checkpoint_dir, fname=f'model_{epoch:05d}')
			trainer.plot(os.path.join(args.vis_dir, f'{plot_name}_train.png'))
			logger.save()

	print(f"STAGE FINISHED! Best epoch = {best_epoch}")


def run(args: Opts):
	device = 'cpu' if args.gpu == -1 else f'cuda:{args.gpu}'
	if args.gpu > -1:
		torch.cuda.set_device(args.gpu)  # Needed to avoid memory access errors for gpu > 0
	assert device == 'cpu' or torch.cuda.is_available(), "CUDA selected but not available."

	specific_feet = None
	if args.two_foot:
		specific_feet = cfg['TWO_FOOT_EXPMT']
	if args.train_on_template:
		specific_feet = cfg['TEMPLATE_FEET']

	dset_kwargs = dict(left_only=args.left_only, tpose_only=args.tpose_only, full_caching=args.full_caching,
					   low_res_textures=args.low_res_textures, low_poly_meshes=args.low_poly_meshes)
	train_dataset = Foot3DDataset(**dset_kwargs, is_train=True, N=args.n_train, train_and_val=args.train_and_val,
								  specific_feet=specific_feet, device=device)
	val_dataset = Foot3DDataset(**dset_kwargs, is_train=False, specific_feet=specific_feet, device=device, N=args.n_val)

	print(f"Num items: {len(train_dataset)} train, {len(val_dataset)} val.")

	# Single texture model
	collate_fn = BatchCollator(device).collate_batches
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=args.shuffle,
							  collate_fn=collate_fn)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=args.shuffle, collate_fn=collate_fn)

	latent_labels = None
	if args.use_latent_labels:
		latent_labels = train_dataset.get_all_keys()
		for k, v in val_dataset.get_all_keys().items():
			latent_labels[k + '_val'] = v

	model = ModelWithLoss(opts=args, device=device,
						  progressive_encoding=args.progressive_encoding,
						  positional_encoding=not args.no_positional_encoding,
						  use_shapevec=True, use_texvec=True, use_posevec=args.use_pose_code,
						  train_size=len(train_dataset), val_size=len(val_dataset), shapevec_size=100, texvec_size=100,
						  posevec_size=100,
						  template_mesh_loc=os.path.join(train_dataset.folder, cfg['TEMPLATE_LOC']),
						  restyle_features_per_vertex=args.restyle_features_per_vertex,
						  restyle_cluster_per_vertex=args.restyle_cluster_per_vertex,
						  latent_labels=latent_labels)

	model = model.to(device)

	optim_network = torch.optim.Adam(model.model.main_params, lr=args.lr_net)
	optim_reg = torch.optim.SGD(model.model.reg_params, lr=args.lr_reg, momentum=0.9)
	optim_val = torch.optim.Adam(model.model.latent_params, lr=args.lr_val)

	if args.latent_optim == 'Adam':
		optim_latent = torch.optim.Adam(model.model.latent_params, lr=args.lr_latent)
	else:
		optim_latent = torch.optim.SGD(model.model.latent_params, lr=args.lr_latent, momentum=0.9)

	out_dir = os.path.join(args.save_dir, args.model_name)
	checkpoint_dir = os.path.join(out_dir, 'checkpoints')
	args.vis_dir = os.path.join(out_dir, 'vis')
	args.render_dir = os.path.join(args.vis_dir, 'renders')

	for d in [out_dir, checkpoint_dir, args.vis_dir, args.render_dir]:
		os.makedirs(d, exist_ok=True)

	# save options
	args.save(os.path.join(out_dir, 'opts.yaml'))

	logger = Logger(os.path.join(out_dir, 'log.txt'), overwrite_log=True)
	sys.stdout = logger

	trainer_reg = Trainer([optim_reg], model, train_loader, val_loader, args,
						  latent_vectors_train=model.model.latent_vectors_train,
						  latent_vectors_val=model.model.latent_vectors_val, val_optim=optim_reg,
						  device=device)  # Just learn template in stage 1
	trainer_net = Trainer([optim_network], model, train_loader, val_loader, args,
						  latent_vectors_train=model.model.latent_vectors_train,
						  latent_vectors_val=model.model.latent_vectors_val, val_optim=optim_val, device=device,
						  n_repeat=args.net_repeat_dataset)
	trainer_latent = Trainer([optim_latent], model, train_loader, val_loader, args,
							 latent_vectors_train=model.model.latent_vectors_train,
							 latent_vectors_val=model.model.latent_vectors_val, val_optim=optim_latent, device=device)

	# Stage 1 - Registration
	if args.reg:
		print("STAGE 1 - REGISTRATION")
		with NoTextureLoading(train_dataset, val_dataset):
			for epoch in range(args.reg_epochs):
				model_kwargs = model_kwargs = dict(chamf=True, smooth=False, gt_z_cutoff=args.gt_z_cutoff)
				msgt = trainer_reg.train_epoch(epoch, model_kwargs=model_kwargs)
				msgv, _ = trainer_reg.val_epoch(epoch, model_kwargs=model_kwargs)
				logger.add_to_log(msgt);
				logger.add_to_log(msgv)
				if (epoch % args.reg_save_every == 0) or (epoch == args.reg_epochs - 1):
					model.save_model(out_dir=out_dir, fname='reg')
					trainer_reg.plot(os.path.join(args.vis_dir, 'reg_train.png'))
					logger.save()

	# STAGE 2 - Network training
	if not args.no_net_train:
		print("STAGE 2 - NETWORK TRAINING")
		train_network(args, model, trainer_net, logger, train_dataset, val_dataset, out_dir=out_dir,
					  checkpoint_dir=checkpoint_dir,
					  num_epochs=args.net_epochs, save_every=args.net_save_every, val_every=args.net_val_every)

	# STAGE 3 - Latent refinement
	if not args.no_latent_refinement:
		print("STAGE 3 - LATENT REFINEMENT")
		train_network(args, model, trainer_latent, logger, train_dataset, val_dataset, out_dir=out_dir,
					  checkpoint_dir=checkpoint_dir,
					  num_epochs=args.latent_epochs, save_every=args.latent_save_every, val_every=args.latent_val_every,
					  plot_name='latent', val_only=True)


if __name__ == '__main__':
	opts = Opts(validate=True)
	run(opts)
