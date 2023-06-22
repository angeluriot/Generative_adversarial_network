import torch
from torch import nn

from training.layers import *
from training.settings import *
from training import utils


# Discriminator block
class DiscriminatorBlock(Module):

	def __init__(self, in_features: int, out_features: int, **kwargs):

		super().__init__(**kwargs)

		self.downsample = Downsampling()
		self.skip = EqualizedConv2D(in_features, out_features, kernel_size = 1, use_bias = False)

		self.layers = nn.Sequential(
			EqualizedConv2D(in_features, in_features, KERNEL_SIZE),
			nn.LeakyReLU(ALPHA),
			EqualizedConv2D(in_features, out_features, KERNEL_SIZE, downsample = True),
			nn.LeakyReLU(ALPHA)
		)

		self.scale = 1.0 / math.sqrt(2.0)


	def forward(self, x: torch.Tensor) -> torch.Tensor:

		skip = self.downsample(x)
		skip = self.skip(skip)

		x = self.layers(x)

		return (x + skip) * self.scale


# Discriminator network
class Discriminator(Module):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)

		self.features_list = utils.get_features(GEN_MIN_FEATURES, GEN_MAX_FEATURES, max_features_first = False)

		self.from_rgb = nn.Sequential(
			EqualizedConv2D(NB_CHANNELS, self.features_list[0], kernel_size = 1),
			nn.LeakyReLU(ALPHA)
		)

		blocks = []

		for i in range(len(self.features_list) - 1):
			blocks.append(DiscriminatorBlock(self.features_list[i], self.features_list[i + 1]))

		self.blocks = nn.Sequential(*blocks)
		self.mini_batch_std = MiniBatchStdDev()

		self.conv = nn.Sequential(
			EqualizedConv2D(self.features_list[-1] + 1, self.features_list[-1], KERNEL_SIZE),
			nn.LeakyReLU(ALPHA)
		)

		self.linear = nn.Sequential(
			EqualizedLinear(self.features_list[-1] * MIN_RESOLUTION * MIN_RESOLUTION, self.features_list[-1]),
			nn.LeakyReLU(ALPHA)
		)

		self.final = EqualizedLinear(self.features_list[-1], 1)


	def forward(self, images: torch.Tensor) -> torch.Tensor:

		x = self.from_rgb(images)
		x = self.blocks(x)

		x = self.mini_batch_std(x)
		x = self.conv(x)
		x = self.linear(x.flatten(1))

		return self.final(x)
