"""Evaluate a texture-neural field trained on a single foot

Trained on '0003/A/0003-A.png'"""
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
from src.model.renderer import FootRenderer
from matplotlib import pyplot as plt
import cv2
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from collections import defaultdict
from tabulate import tabulate
import argparse
from src.train.trainer import sample_latent_vectors
from tqdm import tqdm
from pytorch3d.loss.chamfer import _handle_pointcloud_input, knn_points
from src.vis.mesh_turntable import turntable
from pytorch3d.renderer import TexturesVertex

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

z_cutoff = 0.07

def vis_kps(kps):
	"""KPS shape [K x N x 2].
	Vis all pcls"""
	cols = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
	geom = []
	for n, pts in enumerate(kps):
		geom.append(trimesh.PointCloud(pts, colors=cols[n]))
	sce = trimesh.Scene(geometry=geom)
	sce.show()

def vis_meshes(meshes):
	"""Visualize multiple meshes"""
	cols = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
	geom = []
	for n, mesh in enumerate(meshes):
		tri = trimesh.Trimesh(vertices=mesh.verts_packed().detach().numpy(), faces=mesh.faces_packed().detach().numpy(),
							  process=False)
		tri.visual.face_colors = [*cols[n], 125]
		geom.append(tri)
	sce = trimesh.Scene(geometry=geom)
	sce.show()

def main(src, opts, exp_name=None, out_dir='eval_export/keypoints', gpu=0, no_rendering=False):

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
	imsize = 256
	renderer = FootRenderer(image_size=imsize, device=device)

	collate_fn = BatchCollator(device=device).collate_batches
	dataset = Foot3DDataset(left_only=True, tpose_only=False, is_train=False, device=device, N=1)
	gt_loader = DataLoader(dataset, shuffle=False, collate_fn=collate_fn)

	template_foot = cfg['TEMPLATE_FEET'][0]
	template_dset = Foot3DDataset(left_only=True, tpose_only=True, specific_feet=[template_foot],
								  device=device)

	# For each foot in dataset, get GT keypooints, and validation keypoints
	kp_labels = dataset.keypoint_labels
	gt_kps = torch.zeros((len(dataset), dataset.nkeypoints, 3)).to(device)
	foot_names = []
	gt_meshes = []

	out_data = defaultdict(list)

	def render_correspondences(meshes, keypoints, out_loc):
		# Render feet to images
		R, T = renderer.view_from('topdown')
		out = renderer(meshes, R, T, return_images=True, keypoints=keypoints, keypoints_blend=True)
		imkp = out['keypoints_blend'].cpu().detach().numpy()
		img = (np.hstack(imkp[:, 0]) * 255).astype(np.uint8)
		cv2.imwrite(out_loc, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

	if opts.model_type in ['neural', 'pca']:
		latent_vecs = [model.reg_val, model.shapevec_val]
		if opts.model_type == 'neural':
			latent_vecs.append(model.texvec_val)
		
		if opts.use_pose_code:
			latent_vecs.append(model.posevec_val)
	
	elif opts.model_type == 'supr':
		latent_vecs = [model.betas_val, model.pose_val, model.trans_val, model.reg_val]

	# with NoTextureLoading(dataset, template_dset):
	predictions = []
	for n, batch in enumerate(gt_loader):
		kp_idxs = batch['kp_idxs'].long()
		gt_kps[n] = batch['mesh'].verts_packed()[kp_idxs]
		foot_names.append(batch['name'][0])
		gt_meshes.append(batch['mesh'])

		# Load predicted mesh
		mesh_kwargs = {}
		batch.update(**sample_latent_vectors(batch, latent_vecs))
		# batch['reg_val'][..., 0] += 0.01
		# mesh_kwargs = {k:batch.get(k+'_val', None) for k in ['reg', 'shapevec', 'posevec', 'texvec']}
		# res = model.get_meshes(**mesh_kwargs)
		res = model.get_meshes_from_batch(batch, is_train=False)
		predictions.append(res)


	gt_meshes = join_meshes_as_batch(gt_meshes)
	if not no_rendering:
		render_correspondences(gt_meshes, gt_kps, os.path.join(out_dir, 'gt.png'))

	# Get corresponding keypoints from model predictions
	if opts.model_type == 'neural':
		template_kp_idxs = template_dset[0]['kp_idxs']
	elif opts.model_type == 'pca':
		template_kp_idxs = cfg['PCA_KEYPOINTS']
	elif opts.model_type == 'supr':
		template_kp_idxs = cfg['SUPR_KEYPOINTS']
		
	pred_kps = torch.cat([r['verts'] for r in predictions])[:, template_kp_idxs]
	pred_meshes = join_meshes_as_batch([r['meshes'] for r in predictions])
	if not no_rendering:
		render_correspondences(pred_meshes, pred_kps, os.path.join(out_dir, 'pred.png'))

	# Calculate chamfer losses
	samples = 5000
	gt_pts = sample_points_from_meshes(gt_meshes, num_samples=samples)
	pred_pts = sample_points_from_meshes(pred_meshes, num_samples=samples)
	chamf, _ = chamfer_distance(gt_pts, pred_pts)

	# USE Z CUT-OFF FOR ALTERNATIVE CHAMF LOSS
	chamf_cutoffs = []
	for i in range(gt_pts.shape[0]):
		pred_pts_z_cutoff = pred_pts[i][pred_pts[i, ..., 2] <= z_cutoff]
		gt_pts_z_cutoff = gt_pts[i][gt_pts[i, ..., 2] <= z_cutoff]
		# print(gt_pts_z_cutoff.shape, pred_pts_z_cutoff.shape)
		chamf_cutoffs.append(chamfer_distance(gt_pts_z_cutoff.unsqueeze(0), pred_pts_z_cutoff.unsqueeze(0))[0])

	chamf_z_cutoff = torch.mean(torch.stack(chamf_cutoffs))

	import trimesh
	pcl1 = trimesh.PointCloud(vertices=pred_pts.cpu().detach().numpy()[0], colors=[[255, 0, 0]] * samples)
	pcl2 = trimesh.PointCloud(vertices=gt_pts.cpu().detach().numpy()[0], colors=[[0, 255, 0]] * samples)
	plane = trimesh.Trimesh(vertices=[[-.1, -.1, z_cutoff], [-.1, .1, z_cutoff], [.1, .1, z_cutoff], [.1, -.1, z_cutoff]],
							faces=[[0, 1, 2], [2,3,0]])
	print(1e6*chamf.item(), 1e6*chamf_z_cutoff.item())
	trimesh.Scene([pcl1, pcl2, plane]).show()

	# Render chamfer error spin
	for n in [0]:
		gt_mesh = gt_meshes[n].clone()

		x, x_lengths, _ = _handle_pointcloud_input(pred_meshes[n].verts_padded(), None, None)
		y, y_lengths, _ = _handle_pointcloud_input(gt_mesh.verts_padded(), None, None)
		
		x_sampled, x_sampled_lengths, _ = _handle_pointcloud_input(pred_pts[[n]], None, None)

		x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
		y_nn = knn_points(y, x_sampled, lengths1=y_lengths, lengths2=x_sampled_lengths, K=1) # Has to be to x-sampled to account for low vertex count models (eg SUPR)
		per_vert_errors = x_nn.dists[..., 0]  # (N, P1)

		max_col = 100e-6 #100 um
		col = torch.zeros_like(res['meshes'].verts_padded())
		col[..., 0] = torch.clamp(per_vert_errors / max_col, min=0, max=1)
		res['meshes'].textures = TexturesVertex(col)

		col_gt = torch.zeros_like(gt_mesh.verts_padded())
		col_gt[..., 0] = torch.clamp(y_nn.dists[..., 0] / max_col, min=0, max=1)
		gt_mesh.textures = TexturesVertex(col_gt)

		turntable(res['meshes'], out_loc=os.path.join(out_dir, f'spin_errors_{n:02d}.mp4'), nframes=250)
		turntable(gt_mesh, out_loc=os.path.join(out_dir, f'spin_errors_gt_{n:02d}.mp4'), nframes=250)
	# 	print(x_nn.dists.mean()*1e6, y_nn.dists.mean()*1e6)

	dists = torch.norm(pred_kps - gt_kps, dim=-1)
	out_data['Keypoint (mm)'] = dists.mean().cpu().detach().numpy() * 1e3
	out_data[f'Chamf z-cutoff {z_cutoff} (μm)'] = chamf_z_cutoff.cpu().detach().numpy() * 1e6
	out_data['Chamf (μm)'] = chamf.cpu().detach().numpy() * 1e6

	# PRODUCE PLOT
	fig, ax = plt.subplots()
	ax.imshow(1000*dists.cpu().detach().numpy(), cmap='RdYlGn_r', interpolation='nearest', vmin=0, vmax=10)
	ax.set_xticks(range(len(kp_labels))); ax.set_xticklabels(kp_labels)
	ax.set_yticks(range(len(foot_names))); ax.set_yticklabels(foot_names)

	for i in range(len(foot_names)):
		for j in range(len(kp_labels)):
			text = ax.text(j, i, f'{dists[i, j]*1000:.1f}',
						   ha="center", va="center", color="w")
	fig.savefig(os.path.join(out_dir, 'errors.png'))

	return {k: np.mean(v) for k, v in out_data.items()}

def run_on_exp(exp_name, gpu=0, no_rendering=False):
	"""Given an experiment name, evaluate metrics for all expmts inside"""
	results = []
	with tqdm(os.listdir(os.path.join('exp', exp_name))) as tqdm_it:
		for expmt in tqdm_it:
			tqdm_it.set_description(f"3D evaluating experiment {exp_name}/{expmt}...")
			exp_dir = os.path.join('exp', exp_name, expmt)
			src = os.path.join(exp_dir, 'reg.pth')
			opts = os.path.join(exp_dir, 'opts.yaml')

			out_dir = os.path.join('eval_export', 'eval_3d', exp_name, expmt)

			if not os.path.isfile(src) or not os.path.isfile(opts):
				print(f"Files not found, skipping {expmt}")
				continue
			
			with torch.no_grad():
				res = main(src, opts, out_dir=out_dir, exp_name = expmt, gpu=gpu, no_rendering=no_rendering)
			res['Experiment'] = expmt
			results.append(res)

	# Write to results file
	headers = ['Experiment', 'Keypoint (mm)', 'Chamf (μm)', f'Chamf z-cutoff {z_cutoff} (μm)']
	data = [headers]
	for res in results:
		data.append([res[k] for k in headers])

	results_file = os.path.join('eval_export', 'eval_3d', exp_name, 'results.txt')
	open(results_file, 'w').close() # Clear results file
	with open(results_file, 'a', encoding='UTF-8') as f:
		string = tabulate(data, headers='firstrow', floatfmt=(".1f", ".1f", ".1f"))
		print(string); print(string, file=f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_name', type=str, help="Name of experiment")
	parser.add_argument('--gpu', type=int, default=0, help="GPU ID to use")
	parser.add_argument('--no_rendering', action='store_true', help="Do not render feet")

	args = parser.parse_args()
	run_on_exp(args.exp_name, gpu=args.gpu, no_rendering=args.no_rendering)
