
import torch
from torch import nn

from training.settings import *


# Generator loss
def gen_loss(fake_scores: torch.Tensor) -> torch.Tensor:

	return nn.functional.binary_cross_entropy(fake_scores, torch.ones_like(fake_scores))


# Discriminator fake loss
def disc_loss(fake_scores: torch.Tensor, real_scores: torch.Tensor) -> torch.Tensor:

	fake_loss = nn.functional.binary_cross_entropy(fake_scores, torch.zeros_like(fake_scores))
	real_loss = nn.functional.binary_cross_entropy(real_scores, torch.ones_like(real_scores))

	return fake_loss + real_loss
