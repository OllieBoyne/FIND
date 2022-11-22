"""Load and use style inversion network"""

from multiprocessing.sharedctypes import Value
import torch
import init_paths
import sys
from src.model.restyle_encoder.models.psp import pSp
from argparse import Namespace
nn = torch.nn
from src.model.restyle_encoder.utils.inference_utils import run_on_batch, get_average_image
from src.model.restyle_encoder.options.test_options import TestOptions
import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image

from restyle_encoder.configs.transforms_config import Encode128Transforms
from restyle_encoder.models.stylegan2.model import Generator

from src.model.restyle_cluster_classifier import FeatureClassifier

import math

class TruncatedGenerator(Generator):
	"""Truncated form of StyleGAN2 generator for reduced memory overheads and improved speeds"""
	
	def forward(self, styles, return_latents=False,
			return_features=False, inject_index=None, truncation=1, truncation_latent=None, 
			input_is_latent=False, noise=None, randomize_noise=True, target_feature_maps=None,
			terminate_after_features=True):

		image = None
		if not input_is_latent:
			styles = [self.style(s) for s in styles]

		if noise is None:
			if randomize_noise:
				noise = [None] * self.num_layers
			else:
				noise = [
					getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
				]

		if truncation < 1:
			style_t = []

			for style in styles:
				style_t.append(
					truncation_latent + truncation * (style - truncation_latent)
				)

			styles = style_t

		if len(styles) < 2:
			inject_index = self.n_latent

			if styles[0].ndim < 3:
				latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
			else:
				latent = styles[0]

		else:
			if inject_index is None:
				inject_index = random.randint(1, self.n_latent - 1)

			latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
			latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

			latent = torch.cat([latent, latent2], 1)

		out = self.input(latent)
		out = self.conv1(out, latent[:, 0], noise=noise[0])

		feature_maps = {}

		def add_feature(feat_map, idx):
			if target_feature_maps is None or idx in target_feature_maps:
				feature_maps[idx] = feat_map

		add_feature(out, 0)

		skip = self.to_rgb1(out, latent[:, 1])

		def compose_output():
			res = dict(image=image)
			if return_latents:
				res['latent'] = latent
			if return_features:
				res['features'] = feature_maps
			return res

		i = 1
		for conv1, conv2, noise1, noise2, to_rgb in zip(
				self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
		):
			out = conv1(out, latent[:, i], noise=noise1)
			add_feature(out, i)
			out = conv2(out, latent[:, i + 1], noise=noise2)
			add_feature(out, i+1)
			skip = to_rgb(out, latent[:, i + 2], skip)

			i += 2

			if terminate_after_features and (target_feature_maps is not None and i > max(target_feature_maps)):
				return compose_output() # Terminate early as all feature maps collected

		image = skip

		return compose_output()

class TruncatedpSp(pSp):
	"""Lightweight forward network of Restyle encoder, designed to only store necessary feature maps
	to reduce memory overhead and computation time"""
	def __init__(self, opts):
		super(pSp, self).__init__()
		self.set_opts(opts)
		self.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		# Define architecture
		self.encoder = self.set_encoder()
		self.decoder = TruncatedGenerator(self.opts.output_size, 512, 2, channel_multiplier=1) 
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()

	def forward(self, x, latent=None, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
				inject_latent=None, return_latents=False, alpha=None, average_code=False, input_is_full=False,
				return_features=False, target_feature_maps=None):
		if input_code:
			codes = x
		else:
			codes = self.encoder(x)
			# residual step
			if x.shape[1] == 6 and latent is not None:
				# learn error with respect to previous iteration
				codes = codes + latent
			else:
				# first iteration is with respect to the avg latent code
				codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		if average_code:
			input_is_latent = True
		else:
			input_is_latent = (not input_code) or (input_is_full)

		res = self.decoder([codes], input_is_latent=input_is_latent,
											 randomize_noise=randomize_noise,
											 return_latents=return_latents,
											 return_features=return_features,
											 target_feature_maps=target_feature_maps)

		images = res['image']
		result_latent = res.get('latent', None)
		result_features = res.get('features', None)

		if resize:
			images = self.face_pool(images)

		out = dict(image=images)
		if return_latents: out['latent'] = result_latent
		if return_features: out['features'] = result_features

		return out


class RestyleEncoder(nn.Module):
	def __init__(self, src = 'misc/restyle_encoder_models/rendered_gt_top_views_tpose_128_slice.pt', device='cuda',
						classifier_src=None):
		super().__init__()
		
		# update test options with options used during training
		ckpt = torch.load(src, map_location='cpu')
		opts = ckpt['opts']
		opts.update(dict(resize_outputs=False))
		opts['checkpoint_path'] = src
		self.device = device

		opts = Namespace(**opts)
		opts.device = device
		opts.n_iters_per_batch=2

		self.opts = opts
		self.net = TruncatedpSp(opts).to(device)

		# get the image corresponding to the latent average
		self.avg_image = get_average_image(self.net, self.opts)

		self.transform_np = Encode128Transforms(opts).get_transforms()['transform_inference']
		# self.transform_tensor = torch.jit.script(self.transform)

		self.transform = nn.Sequential(
				transforms.Resize((256, 256)),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

		self.classifier = None
		if classifier_src is not None:
			self.classifier = FeatureClassifier.load('misc/restyle_classifier_models/feat_8_articulated.pth').to(device)
			
		for param in self.parameters():
			param.requires_grad = False

	def __call__(self, images, return_latents=False, return_features=False, target_feature_maps=None, numiter=1):
		"""Receives images of shape (N x C x H x W), returns iterated images and latent vectors"""
		images = self.transform(images.float())

		input_cuda = images.to(self.device).float()
		
		if numiter > 1: raise NotImplementedError("Currently only have single iter Restyle")
		avg_image_for_batch = self.avg_image.unsqueeze(0).repeat(input_cuda.shape[0], 1, 1, 1)
		x_input = torch.cat([input_cuda, avg_image_for_batch], dim=1)

		res = self.net.forward(x_input,
									latent=None,
									randomize_noise=False,
									return_latents=return_latents,
									return_features=return_features,
									target_feature_maps=target_feature_maps,
									resize=self.opts.resize_outputs)

		y_hat = res['image']
		if y_hat is not None:
			y_hat = self.net.face_pool(y_hat)

		res['image'] = y_hat

		if self.classifier:
			if 8 not in target_feature_maps:
				raise NotImplementedError("Classifier currently only works with feature map 8")

			H, W = 128, 128
			_upsample = lambda feat: nn.functional.interpolate(feat, size=(H, W), mode='bilinear')

			feat = (res['features'][8])
			feat = _upsample(feat)

			res['class_logits'] = self.classifier(feat)
			res['class_labels'] = torch.argmax(res['class_logits'], 1)

		return res


def load_img(pth, transform=None):
	from_im = Image.open(pth).convert('RGB')
	if transform:
		from_im = transform(from_im)
	return from_im

def tensor2im(var, norm=True):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	if norm:
		var = ((var + 1) / 2)
		var[var < 0] = 0
		var[var > 1] = 1
	var = var * 255
	return var.astype('uint8')

if __name__ == '__main__':
	from matplotlib import pyplot as plt
	with torch.no_grad():
		net = RestyleEncoder(src='misc/restyle_encoder_models/articulated.pt')

		avg_img = net.avg_image

		# cv2.imwrite('_norm')
		
		#'misc/foot_images/IMG_4902_MASKED.JPG', 
		# img_list = ['misc/foot_images/foot_in_air_MASKED.jpg']
		img_list = ['rendered_gt/rendered_gt_top_views_tpose_128_slice/0000001.png']
		N = len(img_list)
		images = []
		raw_imgs = []
		for n in range(N):
			f = img_list[n]
			# img = np.asarray(Image.open(f).convert('RGB'))
			img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
			img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)[::-1]
			raw_imgs.append(img)
			img = cv2.resize(img, (128, 128))
			images.append(img / 255)
			# images.append(load_img(f, transform=net.transform_np))
		images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
		res = net(images.clone(), return_features=True, target_feature_maps=[8])
		x=0
		# res = net(torch.stack(images))['images']

		for i in range(10):
			np.save('misc/classifier_debug/feat_maps', res['features'][8].cpu().detach().numpy())

		# Show features on a given image
		# for n in range(N):
		# 	M = len(res['images'][0])
		# 	nfeat = 5
		# 	fig, axs = plt.subplots(nrows=1+nfeat, ncols=M+1)
		# 	axs[0, 0].imshow(tensor2im(images[n], norm=False))
		# 	[ax.axis('off') for ax in axs.ravel()]
		# 	for j in range(nfeat):
		# 		axs[j+1, 0].text(.5,.5,f'Feat {j}', fontsize=10,
		# 		bbox={'facecolor':'white','alpha':1,'edgecolor':'none','pad':1},
		# 		ha='center', va='center') 

		# 	for m in range(M):
		# 		ax_img = axs[0, m+1]
		# 		img = tensor2im(res['images'][n][m])
		# 		ax_img.imshow(img)
		# 		ax_img.set_title(f'Iter {m}')

		# 		for j in range(nfeat):
		# 			ax_feat = axs[1+j, m+1]
		# 			img = res['features'][n][m][j].cpu().detach().numpy()
		# 			ax_feat.imshow(img)

		# 	plt.savefig(f'add_in_outputs/{n:03d}.png')


		# Show reconstruction on an image
		# M = 3 # num iters
		# fig, axs = plt.subplots(nrows=N, ncols=M+1)
		# if N == 1 : axs = axs[None, :]
		# for n in range(N):
		# 	axs[n, 0].imshow(tensor2im(images[n], norm=False))
		# 	axs[n,0].axis('off')
		# 	for m in range(M):
		# 		ax = axs[n, m+1]
		# 		ax.axis('off')
		# 		img = tensor2im(res['image'][n])
		# 		ax.imshow(img)
		# 		res = net(res['image'])
		# plt.savefig('_test.png')


		# im2 = tensor2im(res[0][-1])
		# cv2.imwrite('_out1.png', cv2.cvtColor(raw_imgs[0], cv2.COLOR_BGR2RGB))
		# cv2.imwrite('_out2.png', cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
		# print("DONE")