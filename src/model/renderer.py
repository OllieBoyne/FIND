"""Differentiable renderer of foot model"""

from multiprocessing.sharedctypes import Value
import torch
import pytorch3d
from torch import nn
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, FoVPerspectiveCameras,\
	look_at_view_transform,SoftPhongShader, PointLights,SoftSilhouetteShader,BlendParams, TexturesUV,\
 	PointsRasterizationSettings, PointsRenderer, PointsRasterizer,AlphaCompositor, TexturesVertex

from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes

from pytorch3d.renderer.mesh.shader import softmax_rgb_blend
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d import transforms
from pytorch3d.transforms import Transform3d
import numpy as np
import cv2
from typing import Union

def softmax_blend(
	colors: torch.Tensor,
	fragments,
	blend_params: BlendParams,
	znear: Union[float, torch.Tensor] = 1.0,
	zfar: Union[float, torch.Tensor] = 100) -> torch.Tensor:
	"""
	Multi (>3) channel version of softmax_rgb_blend

	Returns:
		RGBA pixel_colors: (N, H, W, 4)
	"""

	N, H, W, K = fragments.pix_to_face.shape
	*_, C = colors.shape
	device = fragments.pix_to_face.device
	pixel_colors = torch.ones((N, H, W, C), dtype=colors.dtype, device=colors.device)
	background_ = blend_params.background_color
	if not isinstance(background_, torch.Tensor):
		background = torch.tensor(background_, dtype=torch.float32, device=device)
	else:
		background = background_.to(device)

	# Weight for background color
	eps = 1e-10

	# Mask for padded pixels.
	mask = fragments.pix_to_face >= 0

	# Sigmoid probability map based on the distance of the pixel to the face.
	prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask
	alpha = torch.prod((1.0 - prob_map), dim=-1)
	# Reshape to be compatible with (N, H, W, K) values in fragments
	if torch.is_tensor(zfar):
		zfar = zfar[:, None, None, None]
	if torch.is_tensor(znear):
		znear = znear[:, None, None, None]

	z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
	z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
	weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)
	delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)
	denom = weights_num.sum(dim=-1)[..., None] + delta

	# Sum: weights * textures + background color
	weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
	weighted_background = delta * background
	pixel_colors = (weighted_colors + weighted_background) / denom

	return pixel_colors

class FeatureShader(nn.Module):
	"""For use for feature maps"""
	
	def __init__(self, device="cpu", blend_params=None):
		super().__init__()
		self.blend_params = blend_params

	def sample_textures(self, meshes: Meshes, fragments, texture=None):
		"""Adapted from meshes.Meshes.sample_textures, to account for a custom texture
		object that isn't meshes.textures"""

		if texture is not None:
			shape_ok = meshes.textures.check_shapes(meshes._N, meshes._V, meshes._F)
			assert shape_ok, "Textures do not match the dimensions of Meshes."

			return texture.sample_textures(fragments, faces_packed=meshes.faces_packed())

		else:
			return meshes.sample_textures(fragments)

	def forward(self, fragments, meshes: Meshes,
				feature_tex:TexturesVertex=None, **kwargs) -> torch.Tensor:
		"""
		Shader for per-vertex features, using softmax blending.
		If feature_tex is given, use this as TexturesVertex object. Otherwise,
		use meshes.textures
		"""
		# get renderer output
		blend_params = kwargs.get('blend_params', self.blend_params)
		texels = self.sample_textures(meshes, fragments, texture=feature_tex)
		images = softmax_blend(texels, fragments, blend_params)
		return images

class FootRenderer(nn.Module):

	def __init__(self, image_size, device='cuda', background_color=(1., 1., 1.), bin_size=None, z_clip_value=None,
			max_faces_per_bin=None):
		super().__init__()

		R, T = look_at_view_transform(dist=0.2)
		self.lights = PointLights(device=device, location=torch.tensor([[0, 0, 100]]))
		self.cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.1) # Placeholder camera

		self.image_size = image_size
		img_blend_params = BlendParams(background_color=background_color)
		self.img_raster_settings = RasterizationSettings(
			image_size=(image_size, image_size), blur_radius=0.,
			faces_per_pixel=1, max_faces_per_bin=max_faces_per_bin,
			bin_size=bin_size, z_clip_value=z_clip_value)

		sil_blend_params = BlendParams(sigma=1e-4)
		self.sil_raster_settings = RasterizationSettings(
			image_size=(image_size, image_size), max_faces_per_bin=max_faces_per_bin,
			blur_radius=np.log(1. / 1e-4 - 1.) * sil_blend_params.sigma, 
			faces_per_pixel=100, bin_size=bin_size)

		# Rasterizers
		self.sil_rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.sil_raster_settings)
		self.img_rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.img_raster_settings)

		# Shaders
		self.img_shader = SoftPhongShader(device=device, cameras=self.cameras, lights=self.lights, blend_params=img_blend_params)
		self.sil_shader = SoftSilhouetteShader()
		self.feature_shader = FeatureShader(device=device)

		# Point cloud rendering
		pcl_raster_settings = PointsRasterizationSettings(image_size=image_size, radius = 0.03, points_per_pixel = 10)
		self.points_rasterizer = PointsRasterizer(cameras=self.cameras, raster_settings=pcl_raster_settings)
		self.points_renderer = PointsRenderer(rasterizer=self.points_rasterizer, compositor=AlphaCompositor())


	def sample_views(self, nviews=1, dist_mean=.25, dist_std=0.05, elev_min=-90, elev_max=90, azim_min=0, azim_max=360,
					seed:int=None):
		if seed:
			np.random.seed(seed)
		distances = np.random.normal(dist_mean, dist_std, (nviews))
		elev = np.random.uniform(elev_min, elev_max, (nviews))
		azim = np.random.uniform(azim_min, azim_max, (nviews))
		R, T = look_at_view_transform(dist=distances, elev=elev, azim=azim, up=((1,0,0),)) # sample random viewing angles and distances
		return R, T


	def linspace_views(self, nviews=1, dist=.3, dist_min=None, dist_max=None,
						 elev_min=None, elev_max=None, azim_min=None, azim_max=None, at=((0, 0, 0),)):

		if dist_min is not None:
			dist = np.linspace(dist_min, dist_max, nviews)
		
		if elev_min is None:
			elev = 0
		else:
			elev = np.linspace(elev_min, elev_max, nviews)

		if azim_min is None:
			azim = 0
		else:
			azim = np.linspace(azim_min, azim_max, nviews)
		R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, up=((1,0,0),), at=at) # sample random viewing angles and distances
		return R, T

	def view_from(self, view_kw='topdown'):
		kws = ['topdown', 'side1', 'side2', 'toes', '45', '60']

		if isinstance(view_kw, str):
			view_kw = [view_kw]

		N = len(view_kw)
		R, T = torch.empty((N, 3, 3)), torch.empty((N, 3))
		for n, v in enumerate(view_kw):
			assert v in kws, f"View description `{view_kw}` not understood"

			dist, elev, azim, point = 0.3, 0, 0, ((0, 0, 0),)
			if v == 'topdown': elev = 0

			if v == 'side1': elev=90; dist=0.35

			if v == 'side2': elev, azim = -90, 180; dist=0.35

			if v == 'toes': point = ((0.1, 0, 0),); dist=0.1

			if v == '45': dist=0.35; elev=-45

			if v == '60': dist=0.35; elev=-60

			_R, _T = look_at_view_transform(dist=dist, elev=elev, azim=azim, up=((1,0,0),), at=point)

			R[n] = _R
			T[n] = _T

		return R, T

	def combine_views(self, R1, T1, R2, T2):
		return torch.cat([R1, R2], dim=0), torch.cat([T1, T2], dim=0)

	def rasterize(self, meshes_world, rasterizers: dict, cameras, **kwargs) -> dict:
		"""Rasterize using multiple rasterizers, sharing transforms.
		Uses a common set of cameras"""
		meshes_proj = rasterizers.get('img', rasterizers.get('sil')).transform(meshes_world, cameras=cameras, **kwargs)
		fragments = {}
		for rname, rasterizer in rasterizers.items():
			raster_settings = rasterizer.raster_settings

			# By default, turn on clip_barycentric_coords if blur_radius > 0.
			# When blur_radius > 0, a face can be matched to a pixel that is outside the
			# face, resulting in negative barycentric coordinates.
			clip_barycentric_coords = raster_settings.clip_barycentric_coords
			if clip_barycentric_coords is None:
				clip_barycentric_coords = raster_settings.blur_radius > 0.0

			# If not specified, infer perspective_correct and z_clip_value from the camera
			if raster_settings.perspective_correct is not None:
				perspective_correct = raster_settings.perspective_correct
			else:
				perspective_correct = cameras.is_perspective()
			if raster_settings.z_clip_value is not None:
				z_clip = raster_settings.z_clip_value
			else:
				znear = cameras.get_znear()
				if isinstance(znear, torch.Tensor):
					znear = znear.min().item()
				z_clip = None if not perspective_correct or znear is None else znear / 2

			pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(meshes_proj,
				image_size=raster_settings.image_size, blur_radius=raster_settings.blur_radius,
				faces_per_pixel=raster_settings.faces_per_pixel, bin_size=raster_settings.bin_size,
				max_faces_per_bin=raster_settings.max_faces_per_bin, clip_barycentric_coords=clip_barycentric_coords,
				perspective_correct=perspective_correct, cull_backfaces=raster_settings.cull_backfaces,
				z_clip_value=z_clip, cull_to_frustum=raster_settings.cull_to_frustum)

			fragments[rname] = Fragments(pix_to_face=pix_to_face, zbuf=zbuf, bary_coords=bary_coords, dists=dists)

		return fragments

	def forward(self, input_meshes: Meshes, R, T,
				return_images=True, return_depth=False, return_mask=False,
				mask_with_grad=True,
				mask_out_faces=False, masked_faces=None,
				keypoints=None, keypoints_blend=False,
				lights = None, return_mask_out_masks=False,
				return_features=False, features=None) -> torch.tensor:
		"""Render a Meshes object, to any of images, depth maps, masks.
		
		input_meshes: N input meshes to render
		R: An (M x 4 x 4) camera rotation for M views
		T: An (M x 4) camera position for M views

		return_mask: Return 0/1 mask of where foot is in image
		mask_with_grad: Compute mask using silhouette renderer to preserve gradients (needed when using masks for loss)

		keypoints: [Optional] (N x K x 3) keypoints to render onto the scene
		keypoints_blend: if keyopints used, return keypoint image blended with renders
		mask_out_faces: Mask out faces either returned by masked_faces, or by looking for u=v=0 in the UV mapping of the mesh textures
		masked_faces: Face indices to mask out. Either a single tensor for all meshes, or a list of N tensors
		"""

		num_views = R.shape[0]
		num_meshes = len(input_meshes)
		meshes = input_meshes.extend(num_views)  # produce a mesh for each view
		R = torch.cat([R]*num_meshes, dim=0)
		T = torch.cat([T]*num_meshes, dim=0)  # produce R, T for each mesh
		cameras = FoVPerspectiveCameras(device=input_meshes.device, R=R, T=T, znear=0.02)

		out_shape_rgb = (num_meshes, num_views, self.image_size, self.image_size, 3)
		out_shape_single = (num_meshes, num_views, self.image_size, self.image_size)

		if lights is None: lights = self.lights

		out = dict()

		# Compute all rasterizations
		rasterizers = {}
		if return_images or return_features:	rasterizers['img'] = self.img_rasterizer
		if return_mask:		rasterizers['sil'] = self.sil_rasterizer
		fragments = self.rasterize(meshes, rasterizers, cameras, lights=lights)

		if return_images:
			renders = self.img_shader(fragments['img'], meshes, cameras=cameras, lights=lights)
			renders = renders[..., :3].reshape(out_shape_rgb)

		if return_features:
			assert features is not None, "Requires features to be passed for rendering"
			feat_tex = TexturesVertex(features).extend(num_views)
			nfeat = features.shape[-1]
			feat_blend_params = BlendParams(background_color=[0.]*nfeat)
			feat_renders = self.feature_shader(fragments['sil'], meshes, feature_tex=feat_tex, blend_params=feat_blend_params)
			feat_renders = feat_renders.reshape(num_meshes, num_views, self.image_size, self.image_size, -1)

		if return_depth:
			if fragments is None: # Only compute fragments if not already rendered
				fragments = self.rasterizer(meshes, cameras=cameras)

			if return_depth:
				out['depth'] = fragments.zbuf.reshape(out_shape_single)

		if return_mask:
			if mask_with_grad or not return_images:
				mask = self.sil_shader(fragments['sil'], meshes, cameras=cameras)[..., 3].reshape(out_shape_single) # Get seg from alpha channel

			else: # hard mask based on image render
				mask = torch.any(renders<1,dim=-1).reshape(out_shape_single).float()


		mask_out_faces_masks = []
		if mask_out_faces: # Mask out faces of images and masks
			# Take only closest face for each pixel. If this face has u=v=0, mask that pixel in the render
			# Shape (N x H x W), as taken only first face per pixel
			pix_to_face = fragments.get('img', fragments.get('sil')).pix_to_face[..., 0]

			# Need to convert pix_to_face to the face count in the specific mesh
			# By default, PyTorch3D counts the face indices cumulatively (eg if mesh 1 has F1 faces, the first face in mesh2 is)
			faces_per_mesh = meshes.num_faces_per_mesh()
			faces_cumulative = torch.cumsum(faces_per_mesh, dim=0) - faces_per_mesh
			pix_to_face = pix_to_face - faces_cumulative.unsqueeze(-1).unsqueeze(-1) 
			pix_to_face = pix_to_face.reshape(num_meshes, num_views, self.image_size, self.image_size) # Reshape to target shape

			textures = input_meshes.textures
			faces_uv, verts_uv = None, None
			if isinstance(textures, TexturesUV):
				faces_uv = textures.faces_uvs_padded()
				verts_uv = textures.verts_uvs_padded()

			for n in range(len(input_meshes)):
				if masked_faces is not None:
					mask_faces = masked_faces[n] if isinstance(masked_faces, list) else masked_faces

				else:
					assert isinstance(meshes.textures, TexturesUV), f"Mask out faces functionality requires masked_faces or TexturesUV - received `{type(meshes.textures)}`"
				
					face_uv = faces_uv[n]
					# if (u, v) = (0, 0) for final UV vertex, that means that some of the faces are set to be masked out. Find and store these
					if (verts_uv[n, -1] == 0).all():
						vt = verts_uv.shape[1] - 1 # index corresponding to this final vertex
						mask_faces = torch.argwhere(torch.all(face_uv==vt, dim=-1)).flatten()
					else:
						break
						
				mask_pix = torch.isin(pix_to_face[n], mask_faces)
				mask_out_faces_masks.append(mask_pix)

				if return_images:
					mpix = mask_pix.unsqueeze(-1).expand(-1, -1, -1, 3)
					renders[n, mpix] = 1.

				if return_mask:
					mpix = mask_pix
					mask[n, mpix] = 0.
				
				if return_features:
					mpix = mask_pix.unsqueeze(-1).expand(-1, -1, -1, feat_renders.shape[-1])
					feat_renders[n, mpix] = 0.

		if keypoints is not None: # Render keypoints onto scene
			n, K, _ = keypoints.shape
			keypoints.reshape(num_meshes, 1, -1, 3).expand(-1, num_views, -1, -1).reshape(-1, K, 3) # Stack in views dimension
			features = torch.zeros_like(keypoints)
			features[..., 0] = 1
			pcl = Pointclouds(points=keypoints, features=features)
			pcl_renders = self.points_renderer(pcl, cameras=cameras).reshape(out_shape_rgb)
			out['keypoints'] = pcl_renders

			if keypoints_blend:
				pcl_mask = torch.any(pcl_renders > 0, dim=-1).unsqueeze(-1)
				out['keypoints_blend'] = ~ pcl_mask * renders.clone() + pcl_mask * pcl_renders

		if return_images: out['image'] = renders
		if return_mask: out['mask'] = mask
		if return_mask_out_masks: out['mask_out_masks'] = torch.cat(mask_out_faces_masks).reshape(out_shape_single)
		if return_features: out['features'] = feat_renders
		
		return out