import torch
from torch import nn

from training.layers import *
from training.settings import *
from training import utils
from training.wavelets import *
import training.augmentation as ada


# Discriminator block
class DiscriminatorBlock(Module):

	def __init__(self, in_features: int, out_features: int, float16: bool = False, **kwargs):

		super().__init__(**kwargs)

		self.dtype = torch.float16 if float16 else torch.float32

		self.layers = nn.Sequential(
			EqualizedConv2D(in_features, in_features, KERNEL_SIZE),
			LeakyReLU(ACTIVATION_GAIN, CLAMP_VALUE),
			EqualizedConv2D(in_features, out_features, KERNEL_SIZE, downsample = True),
			LeakyReLU(1.0, CLAMP_VALUE)
		)

		self.from_wavelets = nn.Sequential(
			EqualizedConv2D(NB_CHANNELS * 4, out_features, kernel_size = 1),
			LeakyReLU(1.0, CLAMP_VALUE)
		)


	def forward(self, x: torch.Tensor, images: torch.Tensor) -> torch.Tensor:

		x = self.layers(x.to(self.dtype))

		return x + self.from_wavelets(images.to(self.dtype))


# Discriminator network
class Discriminator(Module):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)

		self.features_list = utils.get_features(GEN_MIN_FEATURES, GEN_MAX_FEATURES, max_features_first = False)

		self.from_wavelets = nn.Sequential(
			EqualizedConv2D(NB_CHANNELS * 4, self.features_list[1], kernel_size = 1),
			LeakyReLU(ACTIVATION_GAIN, CLAMP_VALUE)
		)

		blocks = []

		for i in range(2, len(self.features_list)):
			blocks.append(DiscriminatorBlock(self.features_list[i - 1], self.features_list[i], i <= NB_FLOAT16_LAYERS + 1))

		self.blocks = nn.ModuleList(blocks)
		self.down_sample = Downsampling()
		self.mini_batch_std = MiniBatchStdDev()

		self.conv = nn.Sequential(
			EqualizedConv2D(self.features_list[-1] + 1, self.features_list[-1], KERNEL_SIZE),
			LeakyReLU(ACTIVATION_GAIN, CLAMP_VALUE)
		)

		self.linear = nn.Sequential(
			EqualizedLinear(self.features_list[-1] * MIN_RESOLUTION * MIN_RESOLUTION, self.features_list[-1]),
			LeakyReLU(ACTIVATION_GAIN)
		)

		self.final = EqualizedLinear(self.features_list[-1], 1)


	def forward(self, images: torch.Tensor, augmentation_proba: float | None = None) -> torch.Tensor:

		if augmentation_proba is not None:
			images = ada.augment(images, augmentation_proba)

		images = discrete_wavelet_transform(images)
		x = self.from_wavelets(images)

		for block in self.blocks:

			images = inverse_wavelet_transform(images)
			images = self.down_sample(images)
			images = discrete_wavelet_transform(images)

			x = block(x, images)

		x = self.mini_batch_std(x.to(torch.float32))
		x = self.conv(x)
		x = self.linear(x.flatten(1))

		return self.final(x)
