from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import perf_counter
import numpy as np
from matplotlib import pyplot as plt
import torch
from src.train.opts import Opts
from src.utils.pytorch3d_tools import to_trimesh
import trimesh
import os
nn = torch.nn

def pretty_print_loss(loss_key:str):
	"""Pretty print a loss string eg loss_def as Def"""
	return loss_key.replace('loss_','').replace('_',' ').title()


def batch_to_device(batch, device='cuda'):
	out = {}
	for k, v in batch.items():
		if hasattr(v, 'to'):
			out[k] = v.to(device)
		else:
			out[k] = v
	return out


def sample_latent_vectors(batch, latent_vectors):
	"""Sample latent vectors with batch. Return as dictionary"""
	if latent_vectors is None:
		return {}		

	out = {}
	for vec in latent_vectors:
		# Access by labels if given
		if vec.labels is not None:
			label_key = vec.key
			assert label_key in batch, f"Trying to sample from latent vector {label_key} using keys, but not found in dataset"
			out[vec.name] = vec[batch[label_key]]

		# Otherwise, just access by index in dataset
		else:
			out[vec.name] = vec[batch['idx']]

	return out

class Trainer:
	def __init__(self, optims, model, train_loader:DataLoader, val_loader:DataLoader, opts:Opts,
				 latent_vectors_train:list=None, latent_vectors_val:list=None, val_optim=None, device='cuda',
				 n_repeat=1):
		"""

		:param optim: either torch Optimizer or list of torch optimizers
		:param model:
		:param data_loader:
		:param train_cfg:
		:param latent_vectors: list of LatentVector objects to be added to batch
		"""

		self.opts = opts
		self.optims = optims if isinstance(optims, list) else [optims]
		self.model = model
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.latent_vectors_train = latent_vectors_train
		self.latent_vectors_val = latent_vectors_val
		self.val_optim = val_optim
		self.device = device

		self.log = defaultdict(dict) # Store per epoch losses for plotting graphs
		self.best_epochs = [] # Store each time a new best epoch is found for plotting graphs
		self.n_repeat = n_repeat

	def sample_latent_vectors(self, batch, latent_vectors=None):
		"""Sample latent vectors with batch. Return as dictionary"""
		if latent_vectors is None:
			latent_vectors = self.latent_vectors_train

		return sample_latent_vectors(batch, latent_vectors)

	def train_epoch(self, epoch, save_model = False, model_kwargs={}):

		epoch_losses = defaultdict(list)
		data_times = []
		net_times = []
		opt_times = []
		model_kwargs['is_train'] = True

		with tqdm(self.train_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as tqdm_it:
			end = perf_counter()
			[o.zero_grad() for o in self.optims]
			for n in range(self.n_repeat):
				for batch in tqdm_it:

					# Add latent vectors to batch
					batch.update(**self.sample_latent_vectors(batch, latent_vectors=self.latent_vectors_train))
					batch = batch_to_device(batch, self.device)

					start = perf_counter()
					data_times.append(start - end)

					if not self.opts.step_per_epoch:
						[o.zero_grad() for o in self.optims]

					loss, loss_dict = self.model(batch, epoch, opts=self.opts, **model_kwargs)

					if loss == 0:
						continue

					end = perf_counter()
					net_times.append(end - start)
					start = perf_counter()

					# Store max GPU usage before loss.backward (as backward() is where graph is freed)
					if self.device == 'cpu':
						a, t = 0, 1e-5
					else:
						a, t = torch.cuda.memory_allocated(self.device), torch.cuda.get_device_properties(self.device).total_memory

					loss.backward()
					if not self.opts.step_per_epoch:
						[o.step() for o in self.optims]

					end = perf_counter()
					opt_times.append(end - start)

					epoch_losses['Loss'].append(loss.item())
					for k, v in loss_dict.items():
						epoch_losses[pretty_print_loss(k)].append(v.item())

					desc = f'[{epoch}] ' + '|'.join([f'{k}:{np.mean(v):.2f}' for k, v in epoch_losses.items()])
					desc += f'| D/N/O: {np.mean(data_times):.2f}/{np.mean(net_times):.2f}/{np.mean(opt_times):.2f}s'
					desc += f' [{100*a/t:.0f}%GPU - {a/1e9:.0f} GB]'
					tqdm_it.set_description('*' * save_model + desc)

			if self.opts.step_per_epoch:
				[o.step() for o in self.optims]

			self.log[epoch]['train_loss'] = {k:v for k, v in epoch_losses.items()}
			msg = tqdm_it.__str__()

		return msg

	def val_epoch(self, epoch, model_kwargs = {}):

		epoch_losses = defaultdict(list)
		model_kwargs['is_train'] = False

		if len(self.val_loader) == 0: return '', {}

		with tqdm(self.val_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as tqdm_it:
			end = perf_counter()
			for batch in tqdm_it:

				# Add latent vectors to batch
				batch.update(**self.sample_latent_vectors(batch, latent_vectors=self.latent_vectors_val))
				batch = batch_to_device(batch, self.device)

				self.val_optim.zero_grad()
				loss, loss_dict = self.model(batch, epoch, opts=self.opts, **model_kwargs)

				loss.backward()
				self.val_optim.step()

				epoch_losses['Loss'].append(loss.item())
				for k, v in loss_dict.items():
					epoch_losses[k.replace('loss_','').lower()].append(v.item())

				tqdm_it.set_description(f'[{epoch} - VAL] ' + '|'.join([f'{pretty_print_loss(k)}:{np.mean(v):.2f}' for k, v in epoch_losses.items()]))

			self.log[epoch]['val_loss'] = {pretty_print_loss(k):v for k, v in epoch_losses.items()}

			msg = tqdm_it.__str__()

		out = {k: np.mean(v) for k, v in epoch_losses.items()}
		return msg, out

	def plot(self, out_loc):
		"""Plot all current losses"""
		
		all_keys = set()
		for e in self.log.values():
			if 'train_loss' in e:
				all_keys |= set(e['train_loss'].keys())
			if 'val_loss' in e:
				all_keys |= set(e['val_loss'].keys())

		all_keys = ['Loss'] + sorted([k for k in all_keys if k!='Loss']) # Make sure overall loss is first

		N = len(all_keys)
		R = int(np.ceil(N/2))
		fig, axs = plt.subplots(ncols=2, nrows=R, figsize=(8, 4*R))
		axs = axs.ravel()
		[ax.axis('off') for ax in axs]

		for n, k in enumerate(all_keys):
			ax = axs[n]
			ax.axis('on')

			ax.set_xlabel('Epoch')
			ax.set_title(k)

			train_x, train_y, val_x, val_y = [], [], [], []

			for epoch in self.log:
				epoch_dict = self.log[epoch]
				if k in epoch_dict.get('train_loss', {}):
					train_x.append(epoch);train_y.append(epoch_dict['train_loss'][k])

				if k in epoch_dict.get('val_loss', {}):
					val_x.append(epoch);val_y.append(epoch_dict['val_loss'][k])

			ax.plot(train_x, [*map(np.mean, train_y)], c='blue')
			ax.plot(val_x, [*map(np.mean, val_y)], c='orange')

			for epoch in self.best_epochs:
				ax.axvline(epoch, alpha=0.3)
		
		fig.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.05)
		fig.tight_layout()
		fig.savefig(out_loc, dpi=200)
		plt.close('all')

	def plot_by_batch(self, out_loc, key):
		"""Given a key, plot only this losses, but split out the batches into individual lines"""
		
		fig, ax = plt.subplots()
		
		ax.set_xlabel('Epoch')
		ax.set_title(key)

		train_x, train_y, val_x, val_y = [], [], [], []
		k = key

		for epoch in self.log:
			epoch_dict = self.log[epoch]
			if k in epoch_dict.get('train_loss', {}):
				train_x.append(epoch);train_y.append(epoch_dict['train_loss'][k])

			if k in epoch_dict.get('val_loss', {}):
				val_x.append(epoch);val_y.append(epoch_dict['val_loss'][k])

		for b in range(len(self.train_loader)):
			ax.plot(train_x, [v[b] for v in train_y], c='blue', alpha=0.2)

		for b in range(len(self.val_loader)):
			ax.plot(val_x, [v[b] for v in val_y], c='orange', alpha=0.2)

		for epoch in self.best_epochs:
			ax.axvline(epoch, alpha=0.3)

		ax.set_title(key)
		
		fig.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.05)
		fig.tight_layout()
		fig.savefig(out_loc, dpi=200)
		plt.close('all')

	def export_meshes(self, export_loc, is_train=False, export_gt=False):

		j = 0
		loader = [self.val_loader, self.train_loader][is_train]
		with tqdm(loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as tqdm_it:
			tqdm_it.set_description(f"Exporting {['val', 'train'][is_train]} meshes")
			for batch in tqdm_it:

				# Add latent vectors to batch
				latent_vectors = [self.latent_vectors_val, self.latent_vectors_train][is_train]
				batch.update(**self.sample_latent_vectors(batch, latent_vectors=latent_vectors))		

				res = self.model.model.get_meshes_from_batch(batch, is_train=is_train)
				
				for i in range(len(res['meshes'])):
					
					mesh = to_trimesh(res['meshes'][i])
					if 'col' in res:
						cols = res['col'][i].cpu().detach().numpy()
						mesh.visual.vertex_colors = cols

					obj_data = trimesh.exchange.obj.export_obj(mesh)
					os.makedirs(export_loc, exist_ok=True)
					with open(os.path.join(export_loc, f"{batch['name'][i]}.obj"), 'w') as outfile:
						outfile.write(obj_data)

					if export_gt:
						vertsgt, facesgt = i['verts'], i['faces']
						meshgt = trimesh.Trimesh(vertices=vertsgt.cpu().detach().numpy(), faces=facesgt.cpu().detach().numpy())

						export_gt_loc = export_loc.replace('meshes', 'meshes_gt')
						obj_data = trimesh.exchange.obj.export_obj(meshgt)
						os.makedirs(export_gt_loc, exist_ok=True)
						with open(os.path.join(export_gt_loc, f"{batch['name'][i]:04d}.obj"), 'w') as outfile:
							outfile.write(obj_data)
					
					j += 1