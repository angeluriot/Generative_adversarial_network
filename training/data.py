import os
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms as transforms

from training.settings import *


# Image dataset class
class ImageDataset(data.Dataset):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)

		self.crop = transforms.Compose([
			transforms.Resize(IMAGE_SIZE, interpolation = transforms.InterpolationMode.LANCZOS),
			transforms.CenterCrop(IMAGE_SIZE)
		])

		self.convert = transforms.ToTensor()

		self.files = []
		files = os.listdir(DATA_DIR)

		for file in files:
			if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png') or file.endswith('.webp') or file.endswith('.bmp') or file.endswith('.tiff'):
				self.files.append(file)


	def __len__(self) -> int:

		return len(self.files) * 2 if FLIP_DATASET else len(self.files)


	def __getitem__(self, index: int | torch.Tensor) -> torch.Tensor:

		if torch.is_tensor(index):
			index = index.int()

		flip = FLIP_DATASET and index >= len(self.files)

		if flip:
			index = index - len(self.files)

		file = os.path.join(DATA_DIR, self.files[index])

		if NB_CHANNELS == 1:
			image = Image.open(file).convert('L')
		elif NB_CHANNELS <= 3:
			image = Image.open(file).convert('RGB')
		else:
			image = Image.open(file).convert('RGBA')

		image = self.crop(image)

		if flip:
			image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

		image = self.convert(image) * 2.0 - 1.0

		if NB_CHANNELS == 2:
			image = image[:2]

		return image


# Final dataset class
class Dataset:

	def __init__(self, dataset: data.Dataset, dataloader: data.DataLoader):

		self.dataset = dataset
		self.dataloader = dataloader
		self.iterator = iter(dataloader)


	# Get the next batch
	def next(self) -> torch.Tensor:

		try:
			return next(self.iterator)
		except StopIteration:
			self.iterator = iter(self.dataloader)
			return next(self.iterator)


	# Get the dataset size
	def size(self) -> int:

		return len(self.dataset)


# Import the dataset
def import_dataset() -> Dataset:

	dataset = ImageDataset()

	dataloader = data.DataLoader(
		dataset,
		batch_size = BATCH_SIZE,
		shuffle = True,
		num_workers = NB_WORKERS,
		pin_memory = True,
		drop_last = True,
		pin_memory_device = DEVICE_NAME if GPU_ENABLED else ''
	)

	return Dataset(dataset, dataloader)
