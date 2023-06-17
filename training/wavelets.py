import torch


# Discrete wavelet transform
def discrete_wavelet_transform(x: torch.Tensor) -> torch.Tensor:

	x = x.permute([0, 2, 3, 1])

	x1 = x[:, 0::2, 0::2, :]
	x2 = x[:, 1::2, 0::2, :]
	x3 = x[:, 0::2, 1::2, :]
	x4 = x[:, 1::2, 1::2, :]

	x_ll = x1 + x2 + x3 + x4
	x_lh = -x1 - x3 + x2 + x4
	x_hl = -x1 + x3 - x2 + x4
	x_hh = x1 - x3 - x2 + x4

	wavelets = torch.cat([x_ll, x_lh, x_hl, x_hh], dim = -1)
	wavelets = wavelets.permute([0, 3, 1, 2])

	return wavelets


# Inverse wavelet transform
def inverse_wavelet_transform(x: torch.Tensor) -> torch.Tensor:

	x = x.permute([0, 2, 3, 1])

	x_ll = x[:, :, :, 0:x.shape[3] // 4]
	x_lh = x[:, :, :, x.shape[3] // 4:x.shape[3] // 4 * 2]
	x_hl = x[:, :, :, x.shape[3] // 4 * 2:x.shape[3] // 4 * 3]
	x_hh = x[:, :, :, x.shape[3] // 4 * 3:]

	x1 = (x_ll - x_lh - x_hl + x_hh) / 4
	x2 = (x_ll - x_lh + x_hl - x_hh) / 4
	x3 = (x_ll + x_lh - x_hl - x_hh) / 4
	x4 = (x_ll + x_lh + x_hl + x_hh) / 4

	y1 = torch.stack([x1, x3], dim = 2)
	y2 = torch.stack([x2, x4], dim = 2)

	recons = torch.cat([y1, y2], dim = -1).reshape((-1, x.shape[1] * 2, x.shape[2] * 2, x.shape[3] // 4))
	recons = recons.permute([0, 3, 1, 2])

	return recons
