import sys

sys.path.append(".")
sys.path.append("..")
# from options.test_options import ClusterOptions
from typing import Tuple
from scipy.cluster.vq import kmeans, kmeans2, whiten
import glob
import numpy as np
import os
from torch import nn
import torch
from tqdm import tqdm
import cv2
from sklearn.cluster import MiniBatchKMeans
from matplotlib import cm
from torchvision import transforms
from PIL import Image
import pickle
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

device = "cuda"

class RestyleFeatures(Dataset):

	def __init__(self,feature_dir: str,image_dir: str,kmeans_model: MiniBatchKMeans,size=(128,128),styles=[8]):
		super(RestyleFeatures,self).__init__()
		self.feature_dir = feature_dir
		self.image_dir = image_dir
		self.kmeans_model = kmeans_model
		self.feature_files = glob.glob(os.path.join(self.feature_dir,"*.npy"))
		self.feature_files.sort()
		self.image_files = [os.path.join(self.image_dir, os.path.splitext(os.path.basename(feature_file))[0]+ ".png") for feature_file in self.feature_files]
		self.toTensor = transforms.ToTensor()
		self.size = size
		self.upsample = nn.UpsamplingBilinear2d(self.size)
		self.styles = styles

	def __getitem__(self, index: int) -> Tuple[torch.Tensor,torch.Tensor]:
		feature_file = self.feature_files[index]
		features = np.load(feature_file,allow_pickle=True)
		img = self.toTensor(Image.open(self.image_files[index]))
		mask = (img.mean(0) < 0.9).float()
		mask = self.upsample(mask.reshape((1,1,mask.shape[0],mask.shape[1]))).squeeze() > 0
		feats = []
		for style in self.styles:
			sampled_feat =  self.upsample(torch.from_numpy(features[style]).unsqueeze(0)).squeeze().numpy()
			sampled_feat = sampled_feat.transpose((1,2,0)).reshape((sampled_feat.shape[1]*sampled_feat.shape[2],-1))
			feats.append(sampled_feat)
		feats = np.hstack(feats)
		labels = self.kmeans_model.predict(feats) + 1
		labels = labels.reshape((self.size[0],self.size[1]))
		labels[mask==False] = 0
		feats = feats.reshape((self.size[0],self.size[1],-1)).transpose((2,0,1))
		return torch.from_numpy(feats).float(), torch.from_numpy(labels).long()
	
	def __len__(self) -> int:
		return len(self.feature_files)

class FeatureClassifier(nn.Module):

	def __init__(self,num_layers=4,input_channels=100, num_classes=21):
		super(FeatureClassifier,self).__init__()

		self.params = dict(num_layers=num_layers, input_channels=input_channels, num_classes=num_classes)

		self.input = nn.Sequential( nn.Dropout(0.1),
									nn.Conv2d(input_channels,512,1),
									nn.LeakyReLU())
		self.layers = nn.ModuleList()
		for n in range(num_layers):
			layer = nn.Sequential( nn.Dropout(0.1),
									nn.Conv2d(int(512/((n+1)**2)),int(512/((n+2)**2)),1),
									nn.LeakyReLU())
			self.layers.append(layer)
		self.logit_layer = nn.Sequential( nn.Dropout(0.1),
									nn.Conv2d(int(512/((n+2)**2)),num_classes,1))
									
		
	def forward(self,feature: torch.Tensor):
		x = self.input(feature)
		for layer in self.layers:
			x = layer(x)
		label_logits = self.logit_layer(x)
		return label_logits

	def predict(self,feature: torch.Tensor):
		logits = self.forward(feature)
		labels = torch.argmax(logits, 1)
		return logits, labels

	def save(self, file):
		data = {'state_dict': self.state_dict(), 'params': self.params}
		torch.save(data, file)

	@classmethod
	def load(cls, file, device='cuda', **kwargs):
		data = torch.load(file, map_location=device)
		state_dict = data['state_dict']
		params = data['params']
		model = cls(**params)
		model.load_state_dict(state_dict)
		return model
		
		
def makeMask(img,threshold = 0.9):
	mask = img.mean(0)< threshold
	return mask

def overlay(img1,img2,alpha):
	img1 = img1.astype(float)
	img2 = img2.astype(float)
	newimg = img1*(alpha) + img2*(1-alpha)
	newimg = newimg.astype(np.uint8)
	return newimg

def make_label_image(label,num_labels,size):
	colormap = cm.get_cmap('viridis', num_labels)
	img = np.ones((size[0]*size[1],3))
	for (i,l) in enumerate(label.reshape(-1)):
		img[i] = img[i] * colormap.colors[l,0:3]
	img = img.reshape((size[0],size[1],3))
	img = img*255.0
	img = img.astype(np.uint8)
	return img
	
def save_label_vis(label,img,output_dir,num_labels,size,id,mask):
	label_img = make_label_image(label,num_labels,size)
	label_img[mask.unsqueeze(2).repeat((1,1,3))==False] = 255
	overlay_img = overlay(img,label_img,0.5)
	filename = os.path.join(output_dir,"%03d.png" % (id))
	cv2.imwrite(filename,label_img)

def save_feature_vis(feature,output_dir, size,id):
	img = feature.astype(float)
	img = np.reshape(img,(size[0],size[1],1))
	img = np.tile(img,(1,1,3))*255.0
	img = img.astype(np.uint8)
	filename = os.path.join(output_dir,"%03d.png" % (id))
	cv2.imwrite(filename,img)

if __name__ == '__main__':
	d = torch.load('misc/restyle_classifier_models/_feat_8_articulated.pth')
	out = {'state_dict': d, 'params': dict(num_layers=4,input_channels=256, num_classes=21)}
	torch.save(out, os.path.join('misc/restyle_classifier_models/feat_8_articulated.pth'))

	# model.save('misc/restyle_classifier_models/feat_8_articulated.pth')


	# feat_maps = torch.from_numpy(np.load('misc/classifier_debug/feat_maps.npy')).cuda()
	
	# logits = feat_classifier(feat_maps)
	# labels = torch.argmax(logits, 1)

	# from matplotlib import pyplot as plt
	# fig, ax = plt.subplots(ncols=5)
	# ax[0].imshow(labels[0].cpu().detach().numpy())
	# for i in range(4):
	#     ax[i+1].imshow(feat_maps[0, i].cpu().detach().numpy())

	# plt.savefig('misc/classifier_debug/vis.png')