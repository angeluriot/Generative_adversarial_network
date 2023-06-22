import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from training.layers import *
from training.settings import *
from training import utils


# Mapping network
class Mapping(Module):

	def __init__(self, epsilon: float = 1e-8, **kwargs):

		super().__init__(**kwargs)

		self.epsilon = float(epsilon)

		layers = []

		for _ in range(MAPPING_LAYERS):
			layers.append(EqualizedLinear(LATENT_DIM, LATENT_DIM, lr_multiplier = MAPPING_LR_RATIO))
			layers.append(nn.LeakyReLU(ALPHA))

		self.layers = nn.Sequential(*layers)


	# Generate a latent vector Z
	@staticmethod
	def gen_z(batch_size: int) -> torch.Tensor:

		return torch.randn((batch_size, LATENT_DIM), device = DEVICE)


	def forward(self, z: torch.Tensor, mean_w: torch.Tensor | None = None, psi: float = 1.0) -> torch.Tensor:

		z = z * (z.square().mean(dim = 1, keepdim = True) + self.epsilon).rsqrt()
		w = self.layers(z)

		if mean_w is not None and psi != 1.0:
			w = mean_w[None, :] + psi * (w - mean_w[None, :])

		return w


# Style block
class StyleBlock(Module):

	def __init__(self, in_features: int, out_features: int, upsample: bool = False, **kwargs):

		super().__init__(**kwargs)

		self.to_style = EqualizedLinear(LATENT_DIM, in_features, bias_init = 1.0)
		self.modulated_conv = ModulatedConv2D(in_features, out_features, KERNEL_SIZE, demodulate = True, upsample = upsample)
		self.noise_scale = nn.Parameter(torch.zeros(()))
		self.bias = nn.Parameter(torch.zeros((out_features,)))
		self.activation = nn.LeakyReLU(ALPHA)


	# Generate noise
	@staticmethod
	def gen_noise(batch_size: int, image_size: int) -> torch.Tensor:

		return torch.randn((batch_size, 1, image_size, image_size), device = DEVICE)


	def forward(self, x: torch.Tensor, w: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:

		style = self.to_style(w)
		x = self.modulated_conv(x, style)

		if noise is None:
			noise = self.gen_noise(x.shape[0], x.shape[2])

		x = x + self.noise_scale * noise
		x = x + self.bias[None, :, None, None]

		return self.activation(x)


# To RGB block
class ToRGB(Module):

	def __init__(self, in_features: int, **kwargs):

		super().__init__(**kwargs)

		self.to_style = EqualizedLinear(LATENT_DIM, in_features, bias_init = 1.0)
		self.modulated_conv = ModulatedConv2D(in_features, NB_CHANNELS, kernel_size = 1, demodulate = False)
		self.bias = nn.Parameter(torch.zeros((NB_CHANNELS,)))


	def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:

		style = self.to_style(w)
		x = self.modulated_conv(x, style)

		return x + self.bias[None, :, None, None]


# Synthesis block
class SynthesisBlock(Module):

	def __init__(self, in_features: int, out_features: int, **kwargs):

		super().__init__(**kwargs)

		self.style_block_1 = StyleBlock(in_features, out_features, upsample = True)
		self.style_block_2 = StyleBlock(out_features, out_features)
		self.to_rgb = ToRGB(out_features)


	def forward(self, x: torch.Tensor, w: torch.Tensor, noise: list[torch.Tensor] | None = None) -> torch.Tensor:

		if w.dim() == 2:
			w.repeat((3, 1, 1))

		if noise is None:
			noise = [None, None]

		x = self.style_block_1(x, w[0], noise[0])
		x = self.style_block_2(x, w[1], noise[1])

		images = self.to_rgb(x, w[2])

		return x, images


# Synthesis network
class Synthesis(Module):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)

		self.features_list = utils.get_features(GEN_MIN_FEATURES, GEN_MAX_FEATURES, max_features_first = True)
		self.const_input = nn.Parameter(torch.randn((self.features_list[0], MIN_RESOLUTION, MIN_RESOLUTION)))
		self.style_block = StyleBlock(self.features_list[0], self.features_list[0])
		self.to_rgb = ToRGB(self.features_list[0])
		blocks = []

		for i in range(len(self.features_list) - 1):
			blocks.append(SynthesisBlock(self.features_list[i], self.features_list[i + 1]))

		self.blocks = nn.ModuleList(blocks)
		self.upsample = Upsampling()


	# Generate noise
	@staticmethod
	def gen_noise(batch_size: int) -> list[torch.Tensor]:

		noise = []
		resolution = MIN_RESOLUTION
		noise.append(StyleBlock.gen_noise(batch_size, resolution))

		for _ in range(1, NB_RESOLUTIONS):
			resolution *= 2
			noise.append(StyleBlock.gen_noise(batch_size, resolution))
			noise.append(StyleBlock.gen_noise(batch_size, resolution))

		return noise


	def forward(self, w: torch.Tensor, noise: list[torch.Tensor] | None = None, mean_w: torch.Tensor | None = None, psi: float = 1.0) -> torch.Tensor:

		if w.dim() == 2:

			if mean_w is not None and psi != 1.0:
				w = mean_w[None, :] + psi * (w - mean_w[None, :])

			w = w.repeat((NB_W, 1, 1))

		elif mean_w is not None and psi != 1.0:
			w = mean_w[None, None, :] + psi * (w - mean_w[None, None, :])

		batch_size = w.shape[1]

		if noise is None:
			noise = [None] * NB_NOISE

		# Initial block
		x = self.const_input.repeat((batch_size, 1, 1, 1))
		x = self.style_block(x, w[0], noise[0])
		images = self.to_rgb(x, w[1])

		# Synthesis blocks
		i = 1

		for block in self.blocks:

			x, new_images = block(x, w[i:i + 3], noise[i:i + 2])
			images = self.upsample(images) + new_images

			i += 2

		return images


# Generator network
class Generator(Module):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)

		self.mapping = Mapping()
		self.synthesis = Synthesis()
		self.mean_w = None


	# Generate latent vectors Z
	@staticmethod
	def gen_z(nb_samples: int = 1) -> torch.Tensor:

		return Mapping.gen_z(nb_samples)


	# Get latent vector W from Z
	def z_to_w(self, z: torch.Tensor, psi: float = 1.0):

		if psi != 1.0 and self.mean_w is None:
			self.compute_mean_w()

		with torch.no_grad():

			w = torch.zeros_like(z, device = DEVICE)

			for i in range(0, z.shape[0], MAPPING_BATCH_SIZE):

				size = min(z.shape[0] - i, MAPPING_BATCH_SIZE)
				w[i:i + size] = self.mapping(z[i:i + size], self.mean_w, psi)

		return w


	# Generate latent vectors W
	def gen_w(self, nb_samples: int = 1, psi: float = 1.0) -> torch.Tensor:

		return self.z_to_w(self.gen_z(nb_samples), psi)


	# Compute the center of the W latent space
	def compute_mean_w(self) -> torch.Tensor:

		with torch.no_grad():

			w = self.gen_w(MEAN_W_SAMPLES)
			self.mean_w = w.mean(dim = 0)

		return self.mean_w


	# Get images from latent vector W
	def w_to_images(self, w: torch.Tensor, noise: list[torch.Tensor] | None = None, psi: float = 1.0) -> torch.Tensor:

		if psi != 1.0 and self.mean_w is None:
			self.compute_mean_w()

		with torch.no_grad():

			images = torch.zeros((w.shape[-2], NB_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), device = DEVICE)

			for i in range(0, w.shape[-2], SYNTHESIS_BATCH_SIZE):

				size = min(w.shape[-2] - i, SYNTHESIS_BATCH_SIZE)
				w_i = w[:, i:i + size] if w.dim() == 3 else w[i:i + size]
				n_i = [n[i:i + size] for n in noise] if noise is not None else None
				images[i:i + size] = self.synthesis(w_i, n_i, self.mean_w, psi)

		return images


	# Get images from latent vector Z
	def z_to_images(self, z: torch.Tensor, noise: list[torch.Tensor] | None = None, psi: float = 1.0) -> torch.Tensor:

		return self.w_to_images(self.z_to_w(z, psi), noise)


	# Generate images
	def generate(self, nb_samples: int, psi: float = 1.0) -> npt.NDArray[np.uint8]:

		return utils.denormalize(self.w_to_images(self.gen_w(nb_samples, psi)))


	# Generate a grid of images
	def generate_grid(self, shape: tuple[int, int], psi: float = 1.0) -> npt.NDArray[np.uint8]:

		images = self.w_to_images(self.gen_w(shape[0] * shape[1], psi))
		return utils.create_grid(images, shape)


	# Generate one image
	def generate_one(self, psi: float = 1.0) -> npt.NDArray[np.uint8]:

		return self.generate(1, psi).squeeze(0)


	# Generate noise
	@staticmethod
	def gen_noise(batch_size: int) -> list[torch.Tensor]:

		return Synthesis.gen_noise(batch_size)


	# Mix two latent vectors W
	@staticmethod
	def style_mix(w_1: torch.Tensor, w_2: torch.Tensor, mix_point: int | None = None) -> torch.Tensor:

		if mix_point is None:
			mix_point = int(torch.rand(()).item() * (NB_W - 1)) + 1

		w_1 = w_1.repeat((mix_point, 1, 1))
		w_2 = w_2.repeat((NB_W - mix_point, 1, 1))

		return torch.cat((w_1, w_2), dim = 0)


	def forward(self, batch_size: int, style_mixing: bool = True, return_w = False):

		if style_mixing and torch.rand(()).item() < STYLE_MIX_PROBA:

			w_1 = self.mapping(self.gen_z(batch_size))
			w_2 = self.mapping(self.gen_z(batch_size))

			w = Generator.style_mix(w_1, w_2)

		else:
			w = self.mapping(self.gen_z(batch_size)).repeat((NB_W, 1, 1))

		if return_w:
			return self.synthesis(w), w

		return self.synthesis(w)
