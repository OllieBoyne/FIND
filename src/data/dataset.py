"""Load Foot3D Dataset"""

import init_paths
from torch.utils.data import Dataset
import trimesh
from src.utils.utils import cfg
import json
from typing import List, Dict
import torch
from torch.utils.data import _utils
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj as p3d_load_obj
import os
from collections import defaultdict, namedtuple
import numpy as np
import cv2

nn = torch.nn


def load_obj(loc, skip_materials=True):
	with open(loc, 'rb') as f:
		mesh = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(f, skip_materials=skip_materials))
	return mesh


def collate_batched_meshes(batch: List[Dict]):  # pragma: no cover
	"""Collate batch into PyTorch3D mesh objects. Has support for all texture types
	"""
	if batch is None or len(batch) == 0:
		return None
	collated_dict = {}
	for k in batch[0].keys():
		collated_dict[k] = [d[k] for d in batch]

	if {"verts", "faces"}.issubset(collated_dict.keys()):

		textures = None
		if 'textures' in collated_dict:
			texlist = collated_dict['textures']
			if texlist[0] is not None:
				textures = texlist[0].join_batch(texlist[1:])

		return Meshes(
			verts=collated_dict["verts"],
			faces=collated_dict["faces"],
			textures=textures,
		)

	else:
		return None


class BatchCollator:
	def __init__(self, device='cuda'):
		self.device = device

	def collate_batches(self, batch: List[Dict]):
		non_mesh_batch = [{k: v for k, v in e.items() if k not in ['verts', 'faces', 'textures']} for e in batch]
		collated_dict = _utils.collate.default_collate(non_mesh_batch)

		collated_dict['mesh'] = collate_batched_meshes(batch).to(self.device)
		for k, v in collated_dict.items():
			if torch.is_tensor(v):
				collated_dict[k] = v.to(self.device)
		return collated_dict


class NoTextureLoading:
	"""Context manager for turning off texture loading, eg for registration"""

	def __init__(self, *datasets):
		self.datasets = datasets
		self.states = []

	def __enter__(self, *args):
		for dataset in self.datasets:
			cur_state = dataset._load_texture
			dataset._load_texture = False
			self.states.append(cur_state)  #  Store original states to restore back to when leaving context

	def __exit__(self, *args):
		for n, dataset in enumerate(self.datasets):
			dataset._load_texture = self.states[n]


def get_pose_code(pose_list: list):
	"""Given a list of pose descriptions, return a vector corresponding to the list. Uses info from cfg"""
	lookup = cfg['POSE_VECTOR']
	N = lookup['SIZE']
	vec = np.zeros(N)

	for p in pose_list:
		p = p.replace('Strong ', '')
		for i in range(N):
			if p in lookup[i]:
				if len(lookup[i]) == 1:
					vec[i] = 1
				elif len(lookup[i]) == 2:
					vec[i] = [-1, 1][lookup[i].index(p)]
				else:
					raise ValueError(f"lookup for pose element {i} is not 1 or 2 long.")
				break

		else:
			raise LookupError(f"Pose {p} not found in lookup.")

	return vec


_cache = {}  # External caching object to use if caching enabled (memory intensive)
CachedMesh = namedtuple('CachedMesh', 'verts face_dict props tex_img')


class Foot3DDataset(Dataset):
	def __init__(self, dataset_json=cfg['DATASET_JSON'],
				 N: int = None, tpose_only=False, left_only=True, specific_feet: list = None,
				 full_caching=False, is_train=True, train_and_val=False, device='cuda',
				 low_res_textures=False, low_poly_meshes=False):
		"""

		:param N: Only use first N elements in dataset
		:param tpose_only: Only use tpose elements
		:param left_only: Only use left feet
		:param single_foot: Only use these Foot-ID (used for pose experiments)
		:param full_caching: Cache textures and meshes.
			Warning: Likely to lead to large memory overheads for large datasets. This setting is useful for speeding
			up small datasets for experiments.
			Also does not work with multiple workers for DataLoader
		low_res_textures: Load 1K textures (requires preprocess_dataset.py script to be run on dataset)
		low_poly_meshes: Loads low poly version of dataset
		"""

		super().__init__()

		folder = cfg['DATASET_FOLDER']
		folder = os.path.join(cfg['DATASET_FOLDER'],
							  cfg['DATASET_NAME'] if not low_poly_meshes else cfg['LOWPOLY_DATASET_NAME'])
		self.folder = folder

		with open(dataset_json) as infile:
			data = json.load(infile)
			self.meta = {k: v for k, v in data.items() if k != 'data'}
			self.data = data['data']

		# Skip foot 3 for now as that is used for template
		self.data = [d for d in self.data if d['Foot ID'] not in cfg['TEMPLATE_FEET'] or specific_feet is not None]

		self.is_train = is_train
		if not (train_and_val or specific_feet):
			if is_train:
				self.data = [d for d in self.data if d['Foot ID'] not in cfg['VAL_FEET']]
			else:
				self.data = [d for d in self.data if d['Foot ID'] in cfg['VAL_FEET']]

		if tpose_only:
			self.data = [d for d in self.data if 'T-Pose' in d.get('pose', [])]

		if left_only:
			self.data = [d for d in self.data if d.get('footedness', None) == 'Left']

		if specific_feet:
			self.data = [d for d in self.data if d['Foot ID'] in specific_feet]
			assert len(self.data) > 0, f"No feet found with IDs `{specific_feet}`."

		if N is not None:
			self.data = self.data[:N]

		# self.data = [d for d in self.data if os.path.isfile(os.path.join(self.folder, d['OBJ file']))]

		self.full_caching = full_caching

		self.keypoint_labels = self.meta['keypoint_labels']
		self.nkeypoints = len(self.keypoint_labels)
		self.device = device
		self._load_texture = True
		self.low_res_textures = low_res_textures

	def __len__(self):
		return len(self.data)

	@property
	def foot_ids(self):
		return [ann['Foot ID'] for ann in self.data]

	@property
	def scan_ids(self):
		return [ann['Scan ID'] for ann in self.data]

	def get_keys(self, idx):
		"""Get keys for shape, pose, tex, reg for a given foot"""
		ann = self.data[idx]
		foot_id = ann['Foot ID']
		scan_id = ann['Scan ID']
		name = f"{foot_id}-{scan_id}"

		texkey = shapekey = foot_id  #  These features should be shared across the same foot
		posekey = regkey = name  # These features should be unique to a particular scan

		return {'shape': shapekey, 'pose': posekey, 'tex': texkey, 'reg': regkey}

	def get_all_keys(self):
		"""Return the keys for an entire dataset"""
		out = defaultdict(list)
		for i in range(len(self)):
			for k, v in self.get_keys(i).items():
				if v not in out[k]:
					out[k].append(v)

		return out

	def get_pose_from_model_id(self, model_id):
		"""Given a Model ID, return the corresponding pose label"""
		foot_id, scan_id = model_id.split("-")
		for ann in self.data:
			if ann['Foot ID'] == foot_id and ann['Scan ID'] == scan_id:
				return ann['pose']

		raise NotFoundErr(f"model_id {model_id} not found in dataset.")

	def get_by_id(self, ID):
		idx = [n for n, ann in enumerate(self.data) if ID == f"{ann['Foot ID']}-{ann['Scan ID']}"]
		assert len(idx) == 1, f'{len(idx)} matches found for ID {ID}.'
		return self[idx[0]]

	def __getitem__(self, idx):
		ann: dict = self.data[idx]
		foot_id = ann['Foot ID']
		scan_id = ann['Scan ID']
		name = f"{foot_id}-{scan_id}"

		obj_loc = os.path.join(self.folder, ann['OBJ file'])
		tex_loc = os.path.join(self.folder, ann['PNG file'])

		# Try to load object data from cache, if not load using PyTorch3D
		load_from_cache = self.full_caching and name in _cache  #  If caching mode, and this object has been cached, then load it

		cached_mesh = None
		if load_from_cache:
			cached_mesh = _cache[name]
			# With the exception if the texture has not previously been loaded, but is needed here, load it
			if self._load_texture and cached_mesh.tex_img is None:
				load_from_cache = False  #  Do not load from cache as need to reload to get texture

		if not load_from_cache:
			verts, face_dict, props = p3d_load_obj(obj_loc, device=self.device, load_textures=False)

			# Load texture image separately
			tex_img = None
			if self._load_texture:
				if self.low_res_textures:
					tex_loc = tex_loc.replace('.png', '_1k.png')

				tex_img = cv2.imread(tex_loc)
				tex_img = (torch.from_numpy(cv2.cvtColor(tex_img, cv2.COLOR_BGR2RGB)) / 255).float()

			# Save object data in cache if caching enabled
			cached_mesh = CachedMesh(verts, face_dict, props, tex_img)
			if self.full_caching:
				_cache[name] = cached_mesh

		tex_map = None
		if self._load_texture:
			if cached_mesh.tex_img is not None:
				img = cached_mesh.tex_img.unsqueeze(0).to(self.device)
				face_uvs = cached_mesh.face_dict.textures_idx.unsqueeze(0).to(self.device)
				verts_uvs = cached_mesh.props.verts_uvs.unsqueeze(0).to(self.device)
				tex_map = TexturesUV(img,
									 faces_uvs=face_uvs,
									 verts_uvs=verts_uvs)
			else:
				raise ValueError(f"Could not load texture - {name}.")

		if ann['footedness'] == 'Right':
			# flip vertices in y direction
			verts[..., 1] = - verts[..., 1]

		verts, face_dict = cached_mesh.verts, cached_mesh.face_dict
		# Shift centroid to origin
		shift = torch.mean(verts, dim=0)
		verts = verts - shift

		# Load correspondences if given
		has_keypoints = False
		keypoints = np.zeros(self.nkeypoints)
		if ann['keypoints'] is not None:
			has_keypoints = True
			keypoints = np.array(ann['keypoints'])

		is_tpose = 'T-Pose' in ann.get('pose', [])

		out = {'faces': face_dict.verts_idx, 'verts': verts,
			   'textures': tex_map, 'idx': idx, 'name': name,
			   'has_keypoints': has_keypoints, 'kp_idxs': keypoints, 'is_tpose': is_tpose,
			   'orig_footedness': ann['footedness'], 'pose_descr': ','.join(ann['pose']),
			   'pose_code': get_pose_code(ann['pose']), **self.get_keys(idx)}

		return out
