import math
import numpy as np
import torch
from torch import nn

from training.settings import *


# Base class for all layers
class Module(nn.Module):

	# Give the number of parameters of the module
	def nb_parameters(self) -> int:

		return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters()])


	# Give the number of trainable parameters of the module
	def nb_trainable_parameters(self) -> int:

		return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters() if p.requires_grad])


	# Give the number of non-trainable parameters of the module
	def nb_non_trainable_parameters(self) -> int:

		return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters() if not p.requires_grad])


	# Summarize the module
	def summary(self) -> None:

		print(f'Number of parameters: {self.nb_parameters():,}')
		print(f'Number of trainable parameters: {self.nb_trainable_parameters():,}')
		print(f'Number of non-trainable parameters: {self.nb_non_trainable_parameters():,}')


	# Remove NaNs from the module gradients
	def clean_nan(self) -> None:

		for p in self.parameters():
			if p.grad is not None:
				torch.nan_to_num(p.grad, nan = 0, posinf = 1e5, neginf = -1e5, out = p.grad)


# Equalized learning rate linear layer
class EqualizedLinear(Module):

	def __init__(self, in_features: int, out_features: int, use_bias: bool = True, bias_init: float = 0.0, lr_multiplier: float = 1.0, **kwargs):

		super().__init__(**kwargs)

		self.use_bias = use_bias
		self.lr_multiplier = float(lr_multiplier)

		self.weight = nn.Parameter(torch.randn((out_features, in_features)) / self.lr_multiplier)

		if self.use_bias:
			self.bias = nn.Parameter(torch.full((out_features,), float(bias_init) / self.lr_multiplier))

		self.gain = self.lr_multiplier / math.sqrt(in_features)


	def forward(self, x: torch.Tensor) -> torch.Tensor:

		if self.use_bias:
			return nn.functional.linear(x, self.weight * self.gain, self.bias * self.lr_multiplier)

		return nn.functional.linear(x, self.weight * self.gain)


# Equalized learning rate 2D convolution layer
class EqualizedConv2D(Module):

	def __init__(self, in_features: int, out_features: int, kernel_size: int,
		use_bias: bool = True, bias_init: float = 0.0, lr_multiplier: float = 1.0, **kwargs):

		super().__init__(**kwargs)

		self.use_bias = use_bias
		self.lr_multiplier = float(lr_multiplier)

		self.weight = nn.Parameter(torch.randn((out_features, in_features, kernel_size, kernel_size)) / self.lr_multiplier)

		if self.use_bias:
			self.bias = nn.Parameter(torch.full((out_features,), float(bias_init) / self.lr_multiplier))

		self.gain = self.lr_multiplier / math.sqrt(in_features * kernel_size * kernel_size)


	def forward(self, x: torch.Tensor) -> torch.Tensor:

		if self.use_bias:
			return nn.functional.conv2d(x, self.weight * self.gain, self.bias * self.lr_multiplier, padding = 'same')

		return nn.functional.conv2d(x, self.weight * self.gain, padding = 'same')


# Modulated 2D convolution layer
class ModulatedConv2D(Module):

	def __init__(self, in_features: int, out_features: int, kernel_size: int,
		demodulate: bool = True, epsilon: float = 1e-8, lr_multiplier: float = 1.0, **kwargs):

		super().__init__(**kwargs)

		self.in_features = in_features
		self.kernel_size = kernel_size
		self.demodulate = demodulate
		self.epsilon = float(epsilon)

		self.weight = nn.Parameter(torch.randn((out_features, in_features, kernel_size, kernel_size)) / float(lr_multiplier))
		self.gain = float(lr_multiplier) / math.sqrt(in_features * kernel_size * kernel_size)


	def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:

		batch_size, _, height, width = x.shape

		# Modulate
		weight = (self.weight * self.gain)[None, :, :, :, :]
		weight = weight * style[:, None, :, None, None]

		# Demodulate
		if self.demodulate:
			sigma = (weight.square().sum(dim = [2, 3, 4]) + self.epsilon).rsqrt()
			weight = weight * sigma[:, :, None, None, None]

		# Reshape input and weight
		x = x.reshape((1, -1, height, width))
		weight = weight.reshape((-1, self.in_features, self.kernel_size, self.kernel_size))

		# Convolution
		x = nn.functional.conv2d(x, weight, padding = 'same', groups = batch_size)

		# Reshape output
		return x.reshape(batch_size, -1, height, width)


# Upsampling layer
class Upsampling(Module):

	def __init__(self, scale_factor: int = 2, **kwargs):

		super().__init__(**kwargs)

		self.up = nn.Upsample(scale_factor = scale_factor, mode = 'nearest')

		pad_sizes = [
			int((len(BLUR_FILTER) - 1) / 2),
			int(math.ceil((len(BLUR_FILTER) - 1) / 2)),
			int((len(BLUR_FILTER) - 1) / 2),
			int(math.ceil((len(BLUR_FILTER) - 1) / 2))
		]

		self.padding = nn.ReflectionPad2d(pad_sizes)

		filter = torch.Tensor(BLUR_FILTER, device = DEVICE).to(dtype = torch.float32)
		filter = filter[:, None] * filter[None, :]
		self.filter = filter / filter.sum()


	def forward(self, x: torch.Tensor) -> torch.Tensor:

		x = self.up(x)
		x = self.padding(x)

		in_features = x.shape[1]
		filter = self.filter[None, None, :, :].repeat((in_features, 1, 1, 1))

		return nn.functional.conv2d(x, filter, groups = x.shape[1])


# Downsampling layer
class Downsampling(Module):

	def __init__(self, scale_factor: int = 2, **kwargs):

		super().__init__(**kwargs)

		self.stride = scale_factor

		pad_sizes = [
			int((len(BLUR_FILTER) - 1) / 2),
			int(math.ceil((len(BLUR_FILTER) - 1) / 2)),
			int((len(BLUR_FILTER) - 1) / 2),
			int(math.ceil((len(BLUR_FILTER) - 1) / 2))
		]

		self.padding = nn.ReflectionPad2d(pad_sizes)

		filter = torch.Tensor(BLUR_FILTER, device = DEVICE).to(dtype = torch.float32)
		filter = filter[:, None] * filter[None, :]
		self.filter = filter / filter.sum()


	def forward(self, x: torch.Tensor) -> torch.Tensor:

		x = self.padding(x)

		in_features = x.shape[1]
		filter = self.filter[None, None, :, :].repeat((in_features, 1, 1, 1))

		return nn.functional.conv2d(x, filter, stride = self.stride, groups = in_features)


# Minibatch standard deviation layer
class MiniBatchStdDev(Module):

	def __init__(self, epsilon: float = 1e-8, **kwargs):

		super().__init__(**kwargs)

		self.epsilon = float(epsilon)


	def forward(self, x: torch.Tensor) -> torch.Tensor:

		batch_size, channels, height, width = x.shape

		assert batch_size % MINIBATCH_STD_GROUP_SIZE == 0

		std = x.reshape((MINIBATCH_STD_GROUP_SIZE, -1, channels, height, width))
		std = std - std.mean(dim = 0)
		std = std.square().mean(dim = 0)
		std = (std + self.epsilon).sqrt()
		std = std.mean(dim = [1, 2, 3])
		std = std.reshape((-1, 1, 1, 1))
		std = std.repeat((MINIBATCH_STD_GROUP_SIZE, 1, height, width))

		return torch.cat([x, std], dim = 1)
