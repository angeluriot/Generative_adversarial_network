import random, platform, psutil
import datetime as dt
import numpy as np
import numpy.typing as npt
import torch
import torchvision.utils as vutils

from training.settings import *


# Reset the random seed
def reset_rand() -> None:

	now = dt.datetime.now()
	seconds_since_midnight = int((now - now.replace(hour = 0, minute = 0, second = 0, microsecond = 0)).total_seconds())
	random.seed(seconds_since_midnight)
	np.random.seed(seconds_since_midnight)
	torch.manual_seed(seconds_since_midnight)


# Check if there is a GPU available
def check_gpu() -> None:

	if GPU_ENABLED:
		torch.cuda.empty_cache()
		nb_gpu = torch.cuda.device_count()
		memory = torch.cuda.mem_get_info()[0] / 1024 ** 3
		print(f'{nb_gpu} GPU {"are" if nb_gpu > 1 else "is"} available! Using GPU: "{torch.cuda.get_device_name()}" ({memory:.2f} GB available)')

	else:
		memory = psutil.virtual_memory().available / 1024 ** 3
		print(f'No GPU available... Using CPU: "{platform.processor()}" ({memory:.2f} GB available)')


# Give the list of features for each layer
def get_features(min_features: int, max_features: int, max_features_first: bool) -> list[int]:

	features_list = []
	features = min_features

	for _ in range(NB_RESOLUTIONS):
		features_list.append(min(features, max_features))
		features *= 2

	if max_features_first:
		return features_list[::-1]

	return features_list


# Denormalize images
def denormalize(images: torch.Tensor) -> npt.NDArray[np.uint8]:

	images = images.clamp(-1, 1)
	images = images.detach().to('cpu').numpy()

	if len(images.shape) == 4:
		images = images.transpose(0, 2, 3, 1)
	else:
		images = images.transpose(1, 2, 0)

	return ((images + 1) * 127.5).astype(np.uint8)


# Create a grid of images from a list of images
def create_grid(images: torch.Tensor, shape: tuple[int, int] | None = None) -> npt.NDArray[np.uint8]:

	if type(images) != torch.Tensor:
		images = torch.as_tensor(images, dtype = torch.float32, device = DEVICE)

	grid = vutils.make_grid(
		images,
		nrow = math.ceil(math.sqrt(BATCH_SIZE)) if shape is None else shape[0],
		padding = MARGIN,
		pad_value = 1.0
	)

	return denormalize(grid)


# Compute the current value from a dictionary
def get_dict_value(dictionary: dict[int | float, float], step: int | float) -> float:

	keys = sorted(list(dictionary.keys()))

	if step < keys[0]:
		return dictionary[keys[0]]

	for i in range(1, len(keys)):
		if step < keys[i]:
			lerp = (step - keys[i - 1]) / (keys[i] - keys[i - 1])
			return (1 - lerp) * dictionary[keys[i - 1]] + lerp * dictionary[keys[i]]

	return dictionary[keys[-1]]
