import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from training.layers import *
from training.settings import *
from training import utils
from training.wavelets import *


# Discriminator network
class Discriminator(Module):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)

		# 64 x 64

		self.init = nn.Sequential(
			Conv2d(NB_CHANNELS, 64, KERNEL_SIZE, stride = 2, padding = 1, padding_mode = 'reflect', bias = False),
			nn.LeakyReLU(ALPHA)
		)

		# 32 x 32

		self.block_1 = nn.Sequential(
			Conv2d(64, 128, KERNEL_SIZE, stride = 2, padding = 1, padding_mode = 'reflect', bias = False),
			BatchNorm2d(128),
			nn.LeakyReLU(ALPHA)
		)

		# 16 x 16

		self.block_2 = nn.Sequential(
			Conv2d(128, 256, KERNEL_SIZE, stride = 2, padding = 1, padding_mode = 'reflect', bias = False),
			BatchNorm2d(256),
			nn.LeakyReLU(ALPHA)
		)

		# 8 x 8

		self.block_3 = nn.Sequential(
			Conv2d(256, 512, KERNEL_SIZE, stride = 2, padding = 1, padding_mode = 'reflect', bias = False),
			BatchNorm2d(512),
			nn.LeakyReLU(ALPHA)
		)

		# 4 x 4

		self.end = nn.Sequential(
			nn.Flatten(),
			Linear(4 * 4 * 512, 1),
			nn.Sigmoid()
		)

		# 1 x 1


	def forward(self, images: torch.Tensor) -> torch.Tensor:

		x = self.init(images)

		x = self.block_1(x)
		x = self.block_2(x)
		x = self.block_3(x)

		return self.end(x)
