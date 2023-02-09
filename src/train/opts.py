import argparse
import yaml
from copy import copy
from src.utils.utils import cfg


def identity(string):
	return string


class Opts():
	restyle_losses = ['restyle_perc_lat', 'restyle_perc_feat', 'restyle_perc_cluster']
	render_losses = ['pix', 'sil', 'vgg_perc', 'restyle_perc_lat', 'restyle_perc_feat', 'restyle_perc_cluster']

	def __init__(self, validate=False):
		"""Read system arguments and parse.
		Set validate = True if only reading system arguments
		(set to False if adding arguments separately, eg from a .yaml file)"""

		p = self.parser = argparse.ArgumentParser()
		p.register('type', None, identity)  # Fix to make ArgumentParser serializable for threading

		# General args
		p.add_argument('--gpu', type=int, default=0, help="CUDA GPU to use (-1 = CPU)")
		p.add_argument('--silent', action='store_true', help="Do not print progress bars to stdout")

		# Dataset args
		p.add_argument('--left_only', action='store_true', help='Only run on left feet')
		p.add_argument('--tpose_only', action='store_true', help='Only run on Tpose feet')
		p.add_argument('--full_caching', action='store_true', help='Fully cache dataset for faster loading.')
		p.add_argument('--train_and_val', action='store_true', help='Train and val in one dataset (for PCA).')

		p.add_argument('--low_poly_meshes', action='store_true', help="Load low polygon (10-20K faces) meshes")
		p.add_argument('--low_res_textures', action='store_true', help="Load low resolution (1K) textures")

		p.add_argument('--n_train', default=None, type=int, help="Number of training feet. Defaults to None (all feet)")
		p.add_argument('--n_val', default=None, type=int, help="Number of val feet. Defaults to None (all feet)")

		# DataLoader args
		p.add_argument('--batch_size_train', type=int, default=1)
		p.add_argument('--batch_size_val', type=int, default=1)
		p.add_argument('--shuffle', action='store_true', help='Shuffle train & val dataloaders')

		# Model arguments
		p.add_argument('--model_type', choices=['neural', 'pca', 'vertexfeatures', 'supr'], default='neural')
		p.add_argument('--load_model', default='', help="Load previous model file (overrides other model args)")
		p.add_argument('--model_name', default='unnnamed', type=str)

		p.add_argument('--progressive_encoding', action='store_true',
					   help="Use progressive encoding in network (untested)")
		p.add_argument('--no_positional_encoding', action='store_true', help="Turn off positional encoding")

		# Optim arguments
		p.add_argument('--lr_net', type=float, default=5e-5)
		p.add_argument('--lr_reg', type=float, default=1e-5)
		p.add_argument('--lr_val', type=float, default=5e-5)
		p.add_argument('--lr_latent', type=float, default=1e-4)

		## Train arguments
		# REG
		p.add_argument('--reg', action='store_true', help="Run registration stage before training network.")
		p.add_argument('--reg_epochs', type=int, default=2000)
		p.add_argument('--reg_save_every', type=int, default=250)

		# NET
		p.add_argument('--no_net_train', action='store_true', help="Don't train network")
		p.add_argument('--net_epochs', type=int, default=1000)
		p.add_argument('--net_save_every', type=int, default=100)
		p.add_argument('--net_val_every', type=int, default=50)
		p.add_argument('--val_crit', default='chamf', type=str, help="Validation critieria used.")

		# LATENT REFINEMENT
		p.add_argument('--no_latent_refinement', action='store_true', help="Don't refine latents")
		p.add_argument('--latent_epochs', type=int, default=500)
		p.add_argument('--latent_optim', type=str, choices=['SGD', 'Adam'], default='Adam')
		p.add_argument('--latent_save_every', type=int, default=250)
		p.add_argument('--latent_val_every', type=int, default=25)

		p.add_argument('--dont_load_latents', action='store_true',
					   help="Reset latent space of loaded model (for fitting to a new set)")
		p.add_argument('--no_rendering', action='store_true',
					   help="Turn off rendering (used for visualisation) within the training loop")

		# Losses for net training
		p.add_argument('--chamf_loss', action='store_true')
		p.add_argument('--smooth_loss', action='store_true')
		p.add_argument('--texture_loss', action='store_true')
		p.add_argument('--pix_loss', action='store_true')
		p.add_argument('--sil_loss', action='store_true')
		p.add_argument('--vgg_perc_loss', action='store_true')
		p.add_argument('--restyle_perc_lat_loss', action='store_true')
		p.add_argument('--restyle_perc_feat_loss', action='store_true')
		p.add_argument('--restyle_perc_cluster_loss', action='store_true')
		p.add_argument('--cont_pose_loss', action='store_true')

		# Weights for net training
		default_weights = dict(chamf=10000, smooth=1000, pix=1., vgg_perc=0.1,
							   restyle_perc_lat=0.25, restyle_perc_feat=1., tex=1., sil=5., restyle_perc_cluster=1.,
							   cont_pose=1.)
		for k, v in default_weights.items():
			p.add_argument(f'--weight_{k}', type=float, default=v)

		# Restyle args
		p.add_argument('--restyle_encoder_path', type=str,
					   default='misc/restyle_encoder_models/rendered_gt_top_views_tpose_128_slice.pt',
					   help="Path to pretrained Restyle encoder")
		p.add_argument('--restyle_feature_maps', type=int, nargs="*", default=None)
		p.add_argument('--restyle_no_masking', action='store_true',
					   help="Turn off masking for restyle feature map loss")

		p.add_argument('--restyle_features_per_vertex', action='store_true', help="Predict restyle features per vertex")
		p.add_argument('--restyle_cluster_per_vertex', action='store_true',
					   help="Predict restyle cluster ID per vertex")
		p.add_argument('--restyle_classifier_path', type=str,
					   default='misc/restyle_classifier_models/feat_8_articulated.pth',
					   help="Path to torch classifier model")

		# Renderer args
		p.add_argument('--num_views', type=int, default=5, help="Views to render per foot")
		p.add_argument('--special_view_type', default=None, choices=[None, 'topdown', 'sample_arc'],
					   help="Special rendering view options")

		p.add_argument('--copy_over_masking', action='store_true', help="Take masking from GT, apply to predicted")
		p.add_argument('--mask_out_pred', action='store_true',
					   help="Mask out directly pred faces (only works with VertexFeatures model")

		## File management
		p.add_argument('--save_dir', default='models', help="Directory in which --model_name directory will be created")

		p.add_argument('--save_meshes_on_end_only', action='store_true',
					   help="Only save predicted meshes on final part of stage")

		# Misc experiments
		p.add_argument('--two_foot', action='store_true',
					   help="Run experiment where only 2 feet (stored in cfg.yaml) are run")
		p.add_argument('--train_on_template', action='store_true',
					   help="Run experiment where only template foot is run")
		p.add_argument('--restrict_3d_n_train', type=int, default=None,
					   help="Restrict the amount of data which can be used for 3D training.")
		p.add_argument('--restrict_3d_train_key', type=str, default=None,
					   help="Restrict the amount of data which can be used for 3D training, by only using data in a list in src/cfg.")
		p.add_argument('--step_per_epoch', action='store_true',
					   help="Step optimizer each epoch, rather than each batch")
		p.add_argument('--net_repeat_dataset', default=1, help="Repeat train dataset N times within each epoch")
		p.add_argument('--gt_z_cutoff', default=None, help="Enforce z cutoff height of GT mesh")

		# VertexFeatures model
		p.add_argument('--vf_optim_features', action='store_true', help="VF model, allow vertex features to optimise")
		p.add_argument('--vf_optim_deforms', action='store_true', help="VF model, allow vertex deforms to optimise")
		p.add_argument('--vf_optim_reg', action='store_true', help="VF model, allow reg to optimise")

		# misc model options
		p.add_argument('--only_classifier_head', action='store_true', help="Only train classifier head")
		p.add_argument('--template_features_pth', type=str, default=None,
					   help="Load pre-trained per-vertex features from this path")

		p.add_argument('--use_latent_labels', action='store_true',
					   help="Use foot scan IDs to refer to elements of latent vectors [experimental]")
		p.add_argument('--use_pose_code', action='store_true', help="Parameterise pose as well [experimental]")

		# For bulk running
		p.add_argument('--exp_name', type=str, default=None,
					   help="Name of experiment, must exist as cfgs/[exp_name].yaml")

		# Evaluation arguments
		p.add_argument('--no_loading', action='store_true', help="For evaluation - do not load from previous optims")
		p.add_argument('--unseen', action='store_true', help="Evaluate on 'unseen' images")
		p.add_argument('--nounseen', action='store_true', help="LEGACY")

		args, _ = p.parse_known_args()

		self.options_list = list(vars(args).keys())
		self.parse(args)

		if validate:
			self.validate_training()

	def parse(self, args):
		for k, v in vars(args).items():
			self.set_option(k, v)

	def validate_training(self):
		"""Run some checks that the input options are valid"""
		losses = self.net_train_kwargs()

		if not self.no_net_train:
			assert any(losses.values()), "At least one loss must be used for net training!"
			assert losses[self.val_crit], f"Val crit set to {self.val_crit}, but this loss isn't used"

		if any(loss not in losses for loss in self.restyle_losses):
			raise NotImplementedError("2D part loss coming soon")

		if isinstance(self.restyle_feature_maps, int):
			self.restyle_feature_maps = [self.restyle_feature_maps]

		if self.restrict_3d_n_train:
			self.step_per_epoch = True  # As losses vary through batch in this experiment, must step per epoch not per loss

		self.train_3d_on_only = None
		if self.restrict_3d_train_key is not None:
			self.train_3d_on_only = cfg[self.restrict_3d_train_key]
			self.step_per_epoch = True

		assert not (self.restrict_3d_n_train and (
					self.restrict_3d_train_key is not None)), "Both restrict_3d_train and restrict_3d_train_key cannot be used simulatenously"

	def net_train_kwargs(self):
		"""Return a dictionary of True/False for the losses to be fed into the model's forward loop."""
		return dict(chamf=self.chamf_loss, smooth=self.smooth_loss,
					texture=self.texture_loss, pix=self.pix_loss, sil=self.sil_loss,
					vgg_perc=self.vgg_perc_loss,
					restyle_perc_lat=self.restyle_perc_lat_loss,
					restyle_perc_feat=self.restyle_perc_feat_loss,
					restyle_perc_cluster=self.restyle_perc_cluster_loss,
					cont_pose=self.cont_pose_loss)

	def use_restyle(self):
		ntk = self.net_train_kwargs()
		return any(ntk[k] for k in self.restyle_losses)

	def training_requires_render(self, stage='net'):
		"""Return True if rendering is required for any of the losses within the training loop"""
		kwargs = self.train_kwargs(stage=stage)
		for k, v in kwargs.items():
			if k in self.render_losses and v:
				return True
		return False

	def set_option(self, option, value):
		assert option in self.options_list, f"Setting `{option}` not found in options."
		setattr(self, option, value)

	def copy_opts(self) -> 'Opts':
		"""Return a new object of Opts with all of the current options carried over"""
		return copy(self)

	def save(self, loc):
		"""Save opts to file"""
		with open(loc, 'w') as outfile:
			args = {k: getattr(self, k) for k in self.options_list}
			yaml.dump(args, outfile, default_flow_style=False)

	@classmethod
	def load(cls, loc):
		"""Load opts from file"""

		with open(loc) as infile:
			cfg = yaml.load(infile, Loader=yaml.FullLoader)

		opts = cls()

		for option, value in cfg.items():
			opts.set_option(option, value)

		opts.validate_training()
		return opts
