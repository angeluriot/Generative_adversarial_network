
import torch
from torch import nn

from training.settings import *


# Generator loss
def gen_loss(fake_scores: torch.Tensor) -> torch.Tensor:

	return nn.functional.softplus(-fake_scores).mean()


# Path length regularization
def path_length(fake_images: torch.Tensor, w: torch.Tensor, mean_path_length: torch.Tensor) -> torch.Tensor:

	noise = torch.rand_like(fake_images) / IMAGE_SIZE

	grad = torch.autograd.grad(
		outputs = (fake_images * noise).sum(),
		inputs = w,
		create_graph = True,
		only_inputs = True
	)[0]

	path_lengths = grad.square().sum(2).mean(0).sqrt()
	path_mean = mean_path_length.lerp(path_lengths.mean(), PATH_LENGTH_DECAY)
	path_penalty = (path_lengths - path_mean).square().mean()

	return path_penalty * PATH_LENGTH_COEF * PATH_LENGTH_INTERVAL, path_mean


# Discriminator fake loss
def disc_fake_loss(fake_scores: torch.Tensor) -> torch.Tensor:

	return nn.functional.softplus(fake_scores).mean()


# Discriminator real loss
def disc_real_loss(real_scores: torch.Tensor) -> torch.Tensor:

	return nn.functional.softplus(-real_scores).mean()


# Gradient penalty
def gradient_penalty(real_images: torch.Tensor, real_scores: torch.Tensor) -> torch.Tensor:

	gradients = torch.autograd.grad(
		outputs = real_scores.sum(),
		inputs = real_images,
		create_graph = True,
		only_inputs = True
	)[0]

	penalty = gradients.square().sum([1, 2, 3]).mean()

	return penalty * 0.5 * GRADIENT_PENALTY_COEF * GRADIENT_PENALTY_INTERVAL
