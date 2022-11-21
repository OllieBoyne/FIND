"""Script installs PyTorch3D on Windows - bypasses issue of CUB not being installed"""

from urllib import request
import pip
import os
import tarfile
import sys
import torch

# pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py37_cu101_pyt108/download.html
# an error might be that cub/cub.h can't be found
# fix with: export CUB_HOME=$PWD/cub-1.10.0

def get_wheel_loc():
	pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
	version_str="".join([
		f"py3{sys.version_info.minor}_cu",
		torch.version.cuda.replace(".",""),
		f"_pyt{pyt_version_str}"
	])
	print(version_str)

def install_pytorch3d():
	url = 'https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz'
	filename = 'cub.1.10.0.tar.gz'
	request.urlretrieve(url, filename)

	# extract tar file
	tar = tarfile.open(filename, 'r:gz')
	tar.extractall()
	tar.close()
	os.remove('cub.1.10.0.tar.gz')

	os.environ['CUB_HOME'] = os.getcwd()+'/cub-1.10.0'
	print('CUB INSTALLED')
	pip.main(["install", r"git+https://github.com/facebookresearch/pytorch3d.git"])

if __name__ == '__main__':
	install_pytorch3d()
	# get_wheel_loc()