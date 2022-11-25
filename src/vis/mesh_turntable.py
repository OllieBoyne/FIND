"""Turntable video of a fitted mesh"""

import sys, os
import init_paths

from src.model.renderer import FootRenderer
from src.model.model import model_from_opts, process_opts

from src.data.dataset import Foot3DDataset, BatchCollator
from tqdm import tqdm
import cv2
from torch.utils.data import DataLoader
import numpy as np
import imageio
import torch
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.decomposition import PCA
from src.utils.vis import visualize_classes, visualize_classes_argmax
from collections import defaultdict
from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform, look_at_rotation

from pytorch3d.transforms import euler_angles_to_matrix, Transform3d


def turntable(mesh: Meshes, out_loc, image_size=256, gpu=0, nframes=250, fps=25, silent=False, azim=-90, dist=0.3):
	if torch.cuda.is_available():
		torch.cuda.set_device(gpu)
		device = f'cuda:{gpu}'
	else:
		device = 'cpu'

	if len(mesh) > 1:
		mesh = mesh[0]
		print("More than 1 mesh given to turntable - only rendering first mesh...")

	renderer = FootRenderer(image_size=image_size, device=device)

	nviews = nframes
	R, T = renderer.linspace_views(nviews=1, dist = dist, azim_min=azim, azim_max=azim)

	verts = mesh.verts_padded()
	theta = torch.linspace(0, 2*np.pi, nviews)

	with tqdm(np.arange(nviews), disable=silent) as tqdm_it:
		tqdm_it.set_description(f'Rendering...')
		frames = []

		for i in tqdm_it:

			euler = torch.tensor([[0, 0, theta[i]]]).to(device)
			transf = Transform3d(device=euler.device).rotate(euler_angles_to_matrix(euler, 'XYZ'))
			X = transf.transform_points(verts)
			mesh = mesh.update_padded(X)

			rdrs = renderer(mesh, R, T)
			N, V, H, W, _ = rdrs['image'].shape

			rdr = (255*rdrs['image'][0, 0].cpu().detach().numpy()).astype(np.uint8)
			rdr = cv2.rotate(rdr, cv2.ROTATE_180) # fix camera rotation
			frames.append(rdr)

	fps = fps
	fname = out_loc
	imageio.mimwrite(fname, frames, fps = fps)
	if not silent:
		print(f"Video written to {fname}")


def render_dataset(gpu=0):
	"""Render all GT & Val dataset"""
	if torch.cuda.is_available():
		torch.cuda.set_device(gpu)
		device = f'cuda:{gpu}'
	else:
		device = 'cpu'

	out_dir = 'misc_vis/dataset_turntables'
	os.makedirs(out_dir, exist_ok=True)
	collate_fn = BatchCollator(device=device).collate_batches
	dataset = Foot3DDataset(left_only=True, tpose_only=False, is_train=True, N=20, device=device)
	gt_loader = DataLoader(dataset, shuffle=False, collate_fn=collate_fn)

	for i in gt_loader:
		turntable(i['mesh'], out_loc=os.path.join(out_dir, i['name'][0]+'.mp4'), nframes=250,
				  gpu=gpu)

if __name__ == '__main__':
	render_dataset()