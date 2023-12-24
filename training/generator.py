import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from training.layers import *
from training.settings import *
from training import utils
from training.wavelets import *


# Generator network
class Generator(Module):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)

		# 1 x 1

		self.init = nn.Sequential(
			Linear(LATENT_DIM, 4 * 4 * 1024, bias = False),
			BatchNorm1d(4 * 4 * 1024),
			nn.ReLU()
		)

		# 4 x 4

		self.block_1 = nn.Sequential(
			ConvTranspose2d(1024, 512, KERNEL_SIZE, stride = 2, padding = 1, bias = False),
			BatchNorm2d(512),
			nn.ReLU()
		)

		# 8 x 8

		self.block_2 = nn.Sequential(
			ConvTranspose2d(512, 256, KERNEL_SIZE, stride = 2, padding = 1, bias = False),
			BatchNorm2d(256),
			nn.ReLU()
		)

		# 16 x 16

		self.block_3 = nn.Sequential(
			ConvTranspose2d(256, 128, KERNEL_SIZE, stride = 2, padding = 1, bias = False),
			BatchNorm2d(128),
			nn.ReLU()
		)

		# 32 x 32

		self.end = nn.Sequential(
			ConvTranspose2d(128, NB_CHANNELS, KERNEL_SIZE, stride = 2, padding = 1, bias = False),
			nn.Tanh()
		)

		# 64 x 64


	# Generate a latent vector Z
	@staticmethod
	def gen_z(batch_size: int) -> torch.Tensor:

		return torch.randn((batch_size, LATENT_DIM), device = DEVICE)


	# Get images from latent vector Z
	def z_to_images(self, z: torch.Tensor) -> torch.Tensor:

		with torch.no_grad():
			return self.forward(z)


	# Generate images
	def generate(self, nb_samples: int) -> npt.NDArray[np.uint8]:

		return utils.denormalize(self.z_to_images(self.gen_z(nb_samples)))


	# Generate one image
	def generate_one(self) -> npt.NDArray[np.uint8]:

		return self.generate(1).squeeze(0)


	def forward(self, z: torch.Tensor) -> torch.Tensor:

		x = self.init(z).reshape((-1, 1024, 4, 4))

		x = self.block_1(x)
		x = self.block_2(x)
		x = self.block_3(x)

		return self.end(x)
