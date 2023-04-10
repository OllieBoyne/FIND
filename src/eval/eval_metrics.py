import torch
nn = torch.nn

def MSE_masked(p1, p2, mask):
	if mask.shape == p1.shape[:-1]: # mask missing channel dimension
		mask = mask.unsqueeze(-1)
		
	mask = mask.expand_as(p1)
	assert mask.shape == p1.shape, f"Not same shape: Mask = {mask.shape}, p1 = {p1.shape}"

	loss = nn.functional.mse_loss(p1, p2, reduction='none')
	loss = (loss * mask.float()).sum()

	non_zero_elements = mask.sum()
	return loss / non_zero_elements

def PSNR_masked(p1, p2, mask):
	psnr = -10 * torch.log10(MSE_masked(p1, p2, mask))
	return psnr

def MSE(p1, p2):
	return nn.functional.mse_loss(p1, p2)

def PSNR(p1, p2):
	mse = nn.functional.mse_loss(p1, p2)
	psnr = -10 * torch.log10(mse)
	return psnr

def IOU(s1: torch.Tensor, s2: torch.Tensor, reduce='mean'):
	"""Calculate intersection / union of two tensors, of size [(optional: batch) x height x width].
	Assume any dims more than 2 are batch dims.
	"""
	assert s1.shape == s2.shape, "S1 and S2 must have same shape"
	*_, H, W = s1.shape
	s1 = s1.reshape(-1, H, W)
	s2 = s2.reshape(-1, H, W)

	intersection = (s1 * s2).sum(dim=(1,2))
	union = torch.maximum(s1, s2).sum(axis=(1,2))

	iou = intersection / union

	if reduce == 'mean':
		return iou.mean()
	else:
		raise NotImplementedError(f"Reduction method `{reduce}` for IOU not implemented.")