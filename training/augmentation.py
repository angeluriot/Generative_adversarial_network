import torch
from torch import nn
import torchvision.transforms as tv
import torchvision.transforms.functional as tvf

from training.layers import *
from training.settings import *
from training import utils


# Apply Adapative Discriminator Augmentation to the images
def augment(images: torch.Tensor, augmentation_proba: float, trans_max = 0.125, scale_std = 0.2, trans_std = 0.125) -> torch.Tensor:

	if augmentation_proba == 0.0 or (not PIXEL_AUGMENTATION and not GEOMETRIC_AUGMENTATION):
		return images

	augmented_images = images.clone()

	for i in range(images.shape[0]):

		if PIXEL_AUGMENTATION:

			# Horizontal flips
			if not FLIP_DATASET and torch.rand(()).item() < augmentation_proba and torch.rand(()).item() < 0.5:
				augmented_images[i] = torch.flip(images[i], dims = [2])
				images = augmented_images.clone()

			# 90° rotations
			if torch.rand(()).item() < augmentation_proba:
				k = torch.randint(1, 4, ()).item()
				augmented_images[i] = torch.rot90(images[i], k = k, dims = (1, 2))
				images = augmented_images.clone()

			# Integer translations
			if torch.rand(()).item() < augmentation_proba:
				image = tvf.pad(images[i], padding = int(trans_max * IMAGE_SIZE), padding_mode = 'reflect')
				x = torch.randint(0, int(trans_max * IMAGE_SIZE) * 2, ()).item()
				y = torch.randint(0, int(trans_max * IMAGE_SIZE) * 2, ()).item()
				augmented_images[i] = tvf.crop(image, y, x, IMAGE_SIZE, IMAGE_SIZE)
				images = augmented_images.clone()

		if GEOMETRIC_AUGMENTATION:

			# Isotropic scaling
			if torch.rand(()).item() < augmentation_proba:
				image = tvf.pad(images[i], padding = int(IMAGE_SIZE * 0.9), padding_mode = 'reflect')
				scale = 1.0 + torch.randn(()).clamp(-0.9, 0.9).item() * scale_std
				image = nn.functional.interpolate(image.unsqueeze(0), scale_factor = scale, mode = 'bilinear').squeeze(0)
				augmented_images[i] = tvf.center_crop(image, IMAGE_SIZE)
				images = augmented_images.clone()

			# Arbitrary rotations (disabled because of a missing PyTorch implementation)
			if False and torch.rand(()).item() < augmentation_proba:
				image = tvf.pad(images[i], padding = int(IMAGE_SIZE * 0.9), padding_mode = 'reflect')
				angle = torch.rand(()).item() * 360.0
				image = tvf.rotate(image, angle = angle, interpolation = tv.InterpolationMode.BILINEAR)
				augmented_images[i] = tvf.center_crop(image, IMAGE_SIZE)
				images = augmented_images.clone()

			# Anisotropic scaling
			if torch.rand(()).item() < augmentation_proba:
				image = tvf.pad(images[i], padding = int(IMAGE_SIZE * 0.9), padding_mode = 'reflect')
				scale_x = 1.0 + torch.randn(()).clamp(-0.9, 0.9).item() * scale_std
				scale_y = 1.0 + torch.randn(()).clamp(-0.9, 0.9).item() * scale_std
				image = nn.functional.interpolate(image.unsqueeze(0), scale_factor = (scale_y, scale_x), mode = 'bilinear').squeeze(0)
				augmented_images[i] = tvf.center_crop(image, IMAGE_SIZE)
				images = augmented_images.clone()

			# Fractional translations (disabled because of a missing PyTorch implementation)
			if False and torch.rand(()).item() < augmentation_proba:
				image = tvf.pad(images[i], padding = int(IMAGE_SIZE * 0.9), padding_mode = 'reflect')
				x = torch.randn(()).clamp(-0.9, 0.9).item() * trans_std * IMAGE_SIZE
				y = torch.randn(()).clamp(-0.9, 0.9).item() * trans_std * IMAGE_SIZE
				image = tvf.affine(image, angle = 0.0, translate = (x, y), scale = 1.0, shear = 0.0, interpolation = tv.InterpolationMode.BILINEAR)
				augmented_images[i] = tvf.center_crop(image, IMAGE_SIZE)
				images = augmented_images.clone()

	images = augmented_images.clone()

	return images
