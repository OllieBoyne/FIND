"""Render validation feet from multiple views, return PSNR"""

"""Evaluate a texture-neural field trained on a single foot

Trained on '0003/A/0003-A.png'"""
from collections import defaultdict
from concurrent.futures import process
import init_paths
from src.model.model import model_from_opts, process_opts
from torch.utils.data import DataLoader
import os
import torch
from src.utils.utils import cfg
import numpy as np
from src.dataset import Foot3DDataset, NoTextureLoading, BatchCollator
import trimesh
from pytorch3d.structures import join_meshes_as_batch
from pytorch3d.renderer import look_at_view_transform
from src.model.renderer import FootRenderer
from matplotlib import pyplot as plt
import cv2
from tabulate import tabulate
from tqdm import tqdm
import argparse
nn = torch.nn

from src.eval.eval_metrics import IOU, MSE, PSNR, MSE_masked, PSNR_masked

def sample_latent_vectors(batch, latent_vectors):
	"""Sample latent vectors with batch. Return as dictionary"""

	out = {}
	for vec in latent_vectors:
		out[vec.name] = vec.data[batch['idx']]

	return out

def main(src, opts, exp_name=None, out_dir='eval_export/psnr', gpu=0):

	if torch.cuda.is_available():
		torch.cuda.set_device(gpu)
		device=f'cuda:{gpu}'
	else:
		device = 'cpu'

	if exp_name == None:
		exp_name = os.path.split(src)[-1]

	os.makedirs(out_dir, exist_ok=True)

	opts = process_opts(opts, eval=True)
	opts.load_model = src
	opts.device = device
	model = model_from_opts(opts)
	model = model.eval().to(device)
	imsize = 128
	renderer = FootRenderer(image_size=imsize, device=device)

	# Load and sample GT colours
	collate_fn = BatchCollator(device=device).collate_batches
	dataset = Foot3DDataset(left_only=True, tpose_only=False, is_train=False,
							#N=model.params['val_size'],
							device=device)
	gt_loader = DataLoader(dataset, shuffle=False, collate_fn=collate_fn)

	template_foot = cfg['TEMPLATE_FEET'][0]
	template_dset = Foot3DDataset(left_only=True, tpose_only=True, specific_feet=[template_foot],
								  device=device)

	out_data = defaultdict(list)

	nviews = 1
	batch_size = 1

	# R, T = renderer.sample_views(nviews=nviews, dist_mean=0.3, dist_std=0, elev_min=-90, elev_max=90,
	# 	azim_min=-90,azim_max=90, seed=20) # Sample same viewpoints for all models

	R, T = renderer.linspace_views(nviews=nviews, dist=0.3, elev_min=-90, elev_max=90) # Sample linear arc for all models

	with tqdm(gt_loader) as tqdm_it:
		tqdm_it.set_description(exp_name)
		for batch in tqdm_it:
			# RENDER GROUND TRUTH
			batch.update(**sample_latent_vectors(batch, model.latent_vectors_val))
						
			res = model.get_meshes_from_batch(batch, is_train=False)
			for b in range(nviews // batch_size):
				subbatch_R, subbatch_T = R[b*batch_size:b*batch_size+batch_size], T[b*batch_size:b*batch_size+batch_size]
				gt_rdrs = renderer(batch['mesh'], subbatch_R, subbatch_T, return_mask=True, mask_out_faces=True, return_mask_out_masks=True)
				pred_rdrs = renderer(res['meshes'], subbatch_R, subbatch_T, return_mask=True)

				# Apply GT masks to predicted
				pred_rdrs['image'][gt_rdrs['mask_out_masks'].unsqueeze(-1).expand_as(pred_rdrs['image'])] = 1.
				pred_rdrs['mask'][gt_rdrs['mask_out_masks'].expand_as(pred_rdrs['mask'])] = 0.


				pred_mask, gt_mask = (pred_rdrs['mask'] > 0), (gt_rdrs['mask'] > 0)
				common_mask = (gt_mask)*(pred_mask)

				# A

				# B
				# pred_rdrs['image'] = pred_rdrs['image'] * pred_mask.unsqueeze(-1)
				# gt_rdrs['image'] = gt_rdrs['image'] * gt_mask.unsqueeze(-1)

				# C
				# pred_rdrs['image'] = pred_rdrs['image'] * common_mask.unsqueeze(-1)
				# gt_rdrs['image'] = gt_rdrs['image'] * common_mask.unsqueeze(-1)

				*_, H, W, v = gt_rdrs['image'].shape

				mask = torch.maximum(pred_rdrs['mask'], gt_rdrs['mask']) # Union of two meshes

				out_data['MSE'].append(MSE(gt_rdrs['image'], pred_rdrs['image']).cpu().detach().numpy())

				# Three methods of calculating PSNR, differing levels of masking
				out_data['PSNR_A'].append(PSNR(gt_rdrs['image'], pred_rdrs['image']).cpu().detach().numpy())
				out_data['PSNR_B'].append(PSNR(gt_rdrs['image']* gt_mask.unsqueeze(-1), pred_rdrs['image']*pred_mask.unsqueeze(-1)).cpu().detach().numpy())
				out_data['PSNR_C'].append(PSNR(gt_rdrs['image']*common_mask.unsqueeze(-1), pred_rdrs['image']*common_mask.unsqueeze(-1)).cpu().detach().numpy())

				out_data['IOU'].append(IOU(gt_rdrs['mask'], pred_rdrs['mask']).cpu().detach().numpy())
				# out_data['MSE_masked'].append(MSE_masked(gt_rdrs['image'], pred_rdrs['image'], mask).cpu().detach().numpy())
				# out_data['PSNR_masked'].append(PSNR_masked(gt_rdrs['image'], pred_rdrs['image'], mask).cpu().detach().numpy())

			# Visualize and save vis (only for last batch)
			idx = batch['idx'].item()
			gt_stacked = np.vstack(gt_rdrs['image'].reshape(-1, H, W, 3).cpu().detach().numpy())
			pred_stacked = np.vstack(pred_rdrs['image'].reshape(-1, H, W, 3).cpu().detach().numpy())
			
			# Show MSE error
			crit = nn.MSELoss(reduction='none')
			mse_err = crit(torch.from_numpy(gt_stacked), torch.from_numpy(pred_stacked))
			mse_err_img = (mse_err / (mse_err.max())).numpy()

			out_stacked = np.hstack([gt_stacked, pred_stacked, mse_err_img])
			cv2.imwrite(os.path.join(out_dir, f'{idx:03d}.png'), cv2.cvtColor((out_stacked * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

	# RENDER 3 VIEW VISUALISATION FOR ALL VALIDATION MESHES AS WELL
	out_pred = []
	out_gt = []
	torch.cuda.empty_cache()
	HD_renderer = FootRenderer(image_size=256, device=device)

	for batch in gt_loader:
		R, T = HD_renderer.view_from(['topdown', 'side1', 'side2'])

		# RENDER GROUND TRUTH
		batch.update(**sample_latent_vectors(batch, model.latent_vectors_val))
		res = model.get_meshes_from_batch(batch, is_train=False)

		# zoom in on toe keypoint as 4th view
		# Get corresponding keypoints from model predictions
		if opts.model_type == 'neural':
			template_kp_idxs = template_dset[0]['kp_idxs']
		elif opts.model_type == 'pca':
			template_kp_idxs = cfg['PCA_KEYPOINTS']
		else:
			template_kp_idxs = cfg['SUPR_KEYPOINTS']

		kp_labels = dataset.keypoint_labels
		gt_kp = batch['mesh'].verts_packed()[batch['kp_idxs'].long()][0, [kp_labels.index('Big toe')]]
		pred_kp = res['verts'][0][template_kp_idxs][[kp_labels.index('Big toe')]]

		Rpred, Tpred = look_at_view_transform(dist=0.15, elev=0, azim=0, up=((1, 0, 0),), at=pred_kp)
		Rgt, Tgt = look_at_view_transform(dist=0.15, elev=0, azim=0, up=((1, 0, 0),), at=gt_kp)

		gt_rdrs = HD_renderer(batch['mesh'], torch.cat([R, Rgt]), torch.cat([T, Tgt]), return_mask=False, mask_out_faces=True, return_mask_out_masks=True)
		pred_rdrs = HD_renderer(res['meshes'], torch.cat([R, Rpred]), torch.cat([T, Tpred]), return_mask=False)

		*_, H, W, v = gt_rdrs['image'].shape

		# Apply GT masks to predicted
		pred_rdrs['image'][gt_rdrs['mask_out_masks'].unsqueeze(-1).expand_as(pred_rdrs['image'])] = 1.

		gt_stacked = np.hstack(gt_rdrs['image'].reshape(-1, H, W, 3).cpu().detach().numpy())
		pred_stacked = np.hstack(pred_rdrs['image'].reshape(-1, H, W, 3).cpu().detach().numpy())
		
		out_pred.append(pred_stacked)
		out_gt.append(gt_stacked)

	cv2.imwrite(os.path.join(out_dir, f'pred_HD.png'), cv2.cvtColor((np.vstack(out_pred) * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
	cv2.imwrite(os.path.join(out_dir, f'gt_HD.png'), cv2.cvtColor((np.vstack(out_gt) * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))


	return {k: np.mean(v) for k, v in out_data.items()}


def run_on_exp(exp_name, gpu=0):
	"""Given an experiment name, evaluate metrics for all expmts inside"""
	results = []
	for expmt in os.listdir(os.path.join('exp', exp_name)):
		exp_dir = os.path.join('exp', exp_name, expmt)
		src = os.path.join(exp_dir, 'model_best.pth')
		opts = os.path.join(exp_dir, 'opts.yaml')

		out_dir = os.path.join('eval_export', 'eval_2d', exp_name, expmt)
		
		if not os.path.isfile(src) or not os.path.isfile(opts):
			print(f"Files not found, skipping {expmt}")
			continue

		with torch.no_grad():
			res = main(src, opts, out_dir=out_dir, exp_name = expmt, gpu=gpu)
		res['Experiment'] = expmt
		results.append(res)

	# Write to results file
	headers = ['Experiment', 'PSNR_A', 'PSNR_B', 'PSNR_C', 'MSE', 'IOU']
	data = [headers]
	for res in results:
		data.append([res[k] for k in headers])

	results_file = os.path.join('eval_export', 'eval_2d', exp_name, 'results.txt')
	open(results_file, 'w').close() # Clear results file
	with open(results_file, 'a') as f:
		string = tabulate(data, headers='firstrow', floatfmt=(".1f", ".1f", ".1f", ".1f", ".5f", ".3f", ".1f", ".5f"))
		print(string); print(string, file=f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_name', type=str, help="Name of experiment")
	parser.add_argument('--gpu', type=int, default=0, help="GPU to use")


	args = parser.parse_args()
	run_on_exp(args.exp_name, gpu=args.gpu)

