from multiprocessing.sharedctypes import Value
import torch
from torch.nn import functional as F
nn = torch.nn
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.loss.chamfer import *
from pytorch3d.ops.sample_points_from_meshes import _rand_barycentric_coords, packed_to_padded, mesh_face_areas_normals

from src.model.vgg_net import LossNetwork
# if torch.cuda.is_available():
#	from src.model.restyle_encoder_model import RestyleEncoder

import os, cv2

from src.utils.vis import visualize_classes, visualize_classes_argmax
import itertools
import numpy as np


class TextureLossGTSpace(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, model, batch:dict,
				num_samples=1000, shapevec=None, texvec=None, posevec=None) -> torch.Tensor:
		"""
		For all vertices in the GT Meshes, sample the exact points on the GT texture and the Neural colour prediction.
		Compare and return an L2 loss

		:param model: NeuralDisplacementField model
		:param batch: Dictionary containing `mesh`: [N,] PyTorch3D GT Meshes
		:return: L2 Loss
		"""

		mesh_gt = batch['mesh']

		sampled_verts, sampled_gt_colours = sample_points_from_meshes(mesh_gt,
																  num_samples=num_samples,
																	  return_textures=True)

		mask = (sampled_gt_colours<1).any(dim=-1).unsqueeze(-1).expand(-1, -1, 3)

		# sampled_verts = sampled_verts[mask]
		# sampled_gt_colours = sampled_gt_colours[mask]

		texvec = texvec if texvec is not None else batch.get('texvec', None)
		shapevec = shapevec if shapevec is not None else batch.get('shapevec', None)
		posevec = posevec if posevec is not None else batch.get('posevec', None)
		res = model(sampled_verts, texvec=texvec, shapevec=shapevec, posevec=posevec)
		sampled_pred_colours = res['col']

		loss = F.mse_loss(sampled_pred_colours, sampled_gt_colours, reduction = 'none')
		loss = loss * mask # multiply out by mask to disregard white points

		return loss.mean()

class DisplacementLoss(nn.Module):

	def forward(self, model, res, batch, epoch, num_samples=5000, z_cutoff=None, gt_z_cutoff=None):

		gt_samples = sample_points_from_meshes(batch['mesh'], num_samples=num_samples)

		pred = res
		pred_meshes = pred['meshes']
		pred_samples = sample_points_from_meshes(pred_meshes, num_samples=num_samples)

		if z_cutoff is not None:
			pred_verts, gt_verts = [], []
			for b in range(len(pred_samples)):
				pred_verts.append(pred_samples[b][pred_samples[b, :, 2] <= z_cutoff])
				gt_verts.append(gt_samples[b][gt_samples[b, :, 2] <= z_cutoff])
			
			pred_pcls = Pointclouds(pred_verts)
			gt_pcls = Pointclouds(gt_verts)
			chamf_loss, _ = chamfer_distance(pred_pcls, gt_pcls)

		elif gt_z_cutoff is not None:
			gt_verts = []
			for b in range(len(gt_samples)):
				gt_verts.append(gt_samples[b][gt_samples[b, :, 2] <= gt_z_cutoff])
			
			gt_pcls = Pointclouds(gt_verts)
			chamf_loss, _ = chamfer_distance(pred_samples, gt_pcls)

		else:
			chamf_loss, _ = chamfer_distance(pred_samples, gt_samples)

		return dict(loss=chamf_loss)


class MeshSmoothnessLoss(nn.Module):
	def forward(self, meshes: Meshes):
		loss_edge = mesh_edge_loss(meshes)
		loss_laplacian = mesh_laplacian_smoothing(meshes, method="cot")
		loss = 0.1 * loss_laplacian + 10 * loss_edge

		return loss



class PerceptualLoss(nn.Module):
	def __init__(self, device='cuda'):
		super().__init__()
		self.feat = LossNetwork(device=device)
		self.crit = nn.MSELoss()

	def forward(self, pred_renders, gt_renders):

		with torch.no_grad():
			gt_feat = self.feat(gt_renders.permute(0, 3, 1, 2))
		
		pred_feat = self.feat(pred_renders.permute(0, 3, 1, 2))

		loss = 0
		for gf, pf in zip(gt_feat, pred_feat):
			loss += self.crit(gf, pf) / len(gt_feat)

		return loss

class SilhouetteLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.crit = nn.MSELoss()
	
	def forward(self, pred, gt):
		return self.crit(pred, gt)

def norm_along_dim(tens, dim=0):
	"""Normalize along a dimension for all values to lie between 0 and 1"""
	tens = tens - tens.min(dim, keepdim=True)[0]
	tens = tens / tens.max(dim, keepdim=True)[0]
	return tens

class RestylePerceptualLoss(nn.Module):
	def __init__(self, pth='', classifier_pth=None, device='cuda'):
		super().__init__()
		if not torch.cuda.is_available():
			raise ImportError("Cannot import Restyle Encoder as no CUDA found.")
		self.encoder = RestyleEncoder(pth, device=device, classifier_src=classifier_pth).eval()
		self._crit = nn.MSELoss()
		self._cluster_crit = nn.CrossEntropyLoss()
		self._cluster_crit_full = nn.CrossEntropyLoss(reduction='none')
	
	def crit(self, pred, gt, pred_mask=None, gt_mask=None):
		"""Return _crit loss between pred and gt. Premultiply by mask"""
		if pred_mask is not None: pred = pred * pred_mask.squeeze(0).unsqueeze(1)
		if gt_mask is not None: gt = gt * gt_mask.squeeze(0).unsqueeze(1)
		return self._crit(pred, gt)

	def debug_vis(self, pred_rdrs, gt_rdrs, pred_feat, gt_feat, out_dir='restyle_debug', pred_mask=None, gt_mask=None, vis_idxs=None):
		"""Visualize"""
		if pred_mask is not None: pred_feat = pred_feat * pred_mask.squeeze(0).unsqueeze(1)
		if gt_mask is not None: gt_feat = gt_feat * gt_mask.squeeze(0).unsqueeze(1)
		MAX_FEAT = 10 # Number of feature maps to show
		rdr_to_im = lambda arr: (255 * arr.cpu().detach().numpy()).astype(np.uint8)
		def fmap_to_im(fmap):
			im = np.zeros((*fmap.shape, 3))
			im[:] = (((fmap - fmap.min()) / (fmap.max() - fmap.min()))*255).astype(np.uint8)[:, :, None]
			# im[:] = (fmap*255).astype(np.uint8)[:, :, None]
			return im

		if vis_idxs is None:
			vis_idxs = np.arange(MAX_FEAT)

		os.makedirs(out_dir, exist_ok=True)
		H, W = pred_feat.shape[-2:]
		for n, pred_rdr in enumerate(pred_rdrs):
			gt_im = np.hstack([rdr_to_im(gt_rdrs[n]), *[fmap_to_im(g.cpu().detach().numpy()) for g in gt_feat[n, vis_idxs]]])
			pred_im = np.hstack([rdr_to_im(pred_rdr), *[fmap_to_im(g.cpu().detach().numpy()) for g in pred_feat[n, vis_idxs]]])
			out_im = np.vstack([gt_im, pred_im]).astype(np.uint8)

			for m, i in enumerate(vis_idxs):
				xy=(int((m + 1.5) * W), 20)
				out_im = cv2.rectangle(out_im, xy, (xy[0]+50, 0), (0, 0, 0), -1)
				cv2.putText(out_im, f'{i:03d}', xy, cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255),1,2)

			fname = os.path.join(out_dir, f'{n:03d}.png')
			cv2.imwrite(fname, cv2.cvtColor(out_im, cv2.COLOR_BGR2RGB))

	def debug_cluster(self, pred_rdrs, gt_rdrs, pred_labels, gt_labels, out_dir='restyle_debug'):
		"""Visualize"""
		rdr_to_im = lambda arr: (255 * arr.cpu().detach().numpy()).astype(np.uint8)
		def fmap_to_im(fmap):
			im = np.zeros((*fmap.shape, 3))
			im[:] = (((fmap - fmap.min()) / (fmap.max() - fmap.min()))*255).astype(np.uint8)[:, :, None]
			# im[:] = (fmap*255).astype(np.uint8)[:, :, None]
			return im

		os.makedirs(out_dir, exist_ok=True)

		for n, pred_rdr in enumerate(pred_rdrs):
			gt_im = np.hstack([rdr_to_im(gt_rdrs[n]), visualize_classes_argmax(gt_labels[n].cpu().detach().numpy())])
			pred_im = np.hstack([rdr_to_im(pred_rdr), visualize_classes_argmax(pred_labels[n].cpu().detach().numpy())])
			out_im = np.vstack([gt_im, pred_im]).astype(np.uint8)

			fname = os.path.join(out_dir, f'{n:03d}.png')
			cv2.imwrite(fname, cv2.cvtColor(out_im, cv2.COLOR_BGR2RGB))
	

	def forward(self, pred, gt, mode='feat', feature_maps=None, pred_masks=None, gt_masks=None, debug=False, debug_dir=None,
				pred_feat = None, pred_logit = None, return_encodings=False):
		"""Takes pred and gt renders in form (N, H, W, 3). Calculates crit loss on latent vectors.
		
		feature_maps: optional list of indexes to collect for feature maps. If None, take all maps"""

		assert mode in ['feat', 'latent', 'cluster'], f"Mode `{mode}` not understood for Restyle loss."

		*_, H, W, _ = pred.shape
		pred, gt = pred.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)
		if pred_feat is not None:
			pred_feat = pred_feat.permute(0, 3, 1, 2)

		_upsample = lambda feat: nn.functional.interpolate(feat, size=(H, W), mode='bilinear')

		encodings = {}

		if mode == 'feat':
			if pred_feat is None:
				pred_feat = self.encoder(pred, return_features=True, target_feature_maps=feature_maps)['features']
			else:
				pred_feat = {8: pred_feat}
			gt_feat = self.encoder(gt, return_features=True, target_feature_maps=feature_maps)['features']

			if feature_maps is None:
				raise NotImplementedError("Can only use specific feature maps")
				N = len(pred_feat)
				return sum(self.crit(p, g, pred_masks, gt_masks) / N for p, g in zip(pred_feat, gt_feat))

			if feature_maps is not None:
				pred_feat_upsampled = torch.cat([_upsample(pred_feat[f]) for f in feature_maps], dim=1)
				gt_feat_upsampled = torch.cat([_upsample(gt_feat[f]) for f in feature_maps], dim=1)

				# Normalize features per-channel
				pred_feat_upsampled = norm_along_dim(pred_feat_upsampled, dim=1)
				gt_feat_upsampled = norm_along_dim(gt_feat_upsampled, dim=1)

				if debug:
					self.debug_vis(pred.permute(0, 2, 3, 1), gt.permute(0, 2, 3, 1), pred_feat_upsampled, gt_feat_upsampled,
					pred_mask=pred_masks, gt_mask=gt_masks, out_dir=debug_dir, vis_idxs = None)

				return self.crit(pred_feat_upsampled, gt_feat_upsampled, pred_masks, gt_masks)

		if mode == 'latent':
			pred_latent = self.encoder(pred, return_latents=True)['latent']
			gt_latent = self.encoder(gt, return_latents=True)['latent']
			return self.crit(pred_latent, gt_latent)
		

		if mode == 'cluster':
			if pred_logit is None:
				pred_logit = self.encoder(pred, return_features=True, target_feature_maps=feature_maps)['class_logits']
			else:
				pred_logit = pred_logit

			gt_logit = self.encoder(gt, return_features=True, target_feature_maps=feature_maps)['class_logits']

			if feature_maps != [8]:
				raise NotImplementedError("Classifier only works with exactly feat map 8 currently.")

			pred_logit_upsampled = _upsample(pred_logit) # torch.cat([_upsample(pred_feat[f]) for f in feature_maps], dim=1)
			gt_logit_upsampled = _upsample(gt_logit) # torch.cat([_upsample(gt_feat[f]) for f in feature_maps], dim=1)

			gt_labels = torch.argmax(gt_logit_upsampled, dim=1)

			if pred_masks is None:
				raise NotImplementedError("Clustering loss requires masking")

			# Need to set background as class 0 as probability 1 with high probability
			if pred_masks is not None:
				background_probs = (pred_masks == 0) * 100
				pred_logit_upsampled[:, 0] = background_probs

			full_loss = self._cluster_crit_full(pred_logit_upsampled, gt_labels)
			full_loss = full_loss * pred_masks # Apply masking

			if return_encodings:
				encodings['gt_labels'] = gt_labels
				encodings['gt_logits'] = gt_logit_upsampled
				encodings['CE_loss'] = full_loss

			if debug:
				pred_labels = torch.argmax(pred_logit_upsampled, dim=1)
				self.debug_cluster(pred.permute(0, 2, 3, 1), gt.permute(0, 2, 3, 1), pred_labels, gt_labels, out_dir=debug_dir)

				# VISUALISE PROB HEATMAPS
				rdr_to_im = lambda arr: (255 * arr.cpu().detach().numpy()).astype(np.uint8)
				def fmap_to_im(fmap):
					im = np.zeros((*fmap.shape, 3))
					im[:] = (((fmap - fmap.min()) / (fmap.max() - fmap.min()))*255).astype(np.uint8)[:, :, None]
					return im

				hmps = [fmap_to_im(pred_logit_upsampled[0, n].cpu().detach().numpy()) for n in range(5)]
				x = fmap_to_im(pred_labels[0].cpu().detach().numpy())
				im = np.hstack([rdr_to_im(pred[0].permute(1,2,0)), x, *hmps]).astype(np.uint8)

				fname = os.path.join(debug_dir, f'_probs.png')
				cv2.imwrite(fname, cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
				cv2.imwrite(os.path.join(debug_dir, '_CELoss.png'), fmap_to_im(full_loss[0].cpu().detach().numpy()))

			return full_loss.mean(), encodings


class ContrastiveLoss(nn.Module):

	def crit(self, vec1, vec2, code1, code2, margin=0.5):
		"""Given two vectors, vec1, vec2, and corresponding codes code1 and code 2, return a contrastive loss
		L = y d^2 + (1 - y) max(margin-d, 0)^2
		where y is a metric of similarity between code1 and code2
		and d is the distance between vec1 and vec2"""
		y = (code1 * code2).sum()

		d = torch.norm(vec1 - vec2)
		return y * d**2 + (1 - y) * max(margin - d**2, 0) ** 2

	def forward(self, vecs, codes, npairs=10):
		"""Given a set of vectors [N x K], and a set of codes [N x C], select a subset of pairs n, and return the mean
		contrastive loss between the pairs
		"""
		N, K = vecs.shape
		max_pairs = (N * (N - 1))//2  # Equivalent to nCk for k=2
		npairs = min(npairs, max_pairs)

		idxs = np.arange(N)
		all_pairs = list(itertools.permutations(idxs, 2))
		np.random.shuffle(all_pairs)

		loss = []
		for pair in all_pairs[:npairs]:
			loss.append(self.crit(*vecs[[*pair]], *codes[[*pair]]))

		return sum(loss) / npairs  # Average loss per pair
