import sys, random, os, math, threading
import numpy as np
import numpy.typing as npt
from PIL import Image
from pytorch_fid.fid_score import calculate_fid_given_paths

from training.generator import Generator
from training.settings import *


def clone_dataset() -> None:

	files = os.listdir(DATA_DIR)
	random.shuffle(files)

	if Image.open(os.path.join(DATA_DIR, files[0])).size == (IMAGE_SIZE, IMAGE_SIZE):
		return

	if not os.path.exists(FID_REAL_DIR):
		os.makedirs(FID_REAL_DIR)

	if not os.path.exists(FID_FAKE_DIR):
		os.makedirs(FID_FAKE_DIR)

	if len(os.listdir(FID_REAL_DIR)) > 0:
		return

	def process_image(input_path: str, output_path: str) -> None:
		image = Image.open(input_path)
		image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
		image.save(output_path)

	nb = 0
	threads = []

	while nb < FID_NB_SAMPLES:

		for file in files:

			if nb >= FID_NB_SAMPLES:
				break

			t = threading.Thread(
				target = process_image,
				args = (
					os.path.join(DATA_DIR, file),
					os.path.join(FID_REAL_DIR, f'{str(nb).zfill(math.ceil(math.log10(FID_NB_SAMPLES)))}.png')
				)
			)

			t.start()
			threads.append(t)
			nb += 1

	for t in threads:
		t.join()


def compute_fid(generator: Generator) -> float:

	threads = []

	for file in os.listdir(FID_FAKE_DIR):
		t = threading.Thread(target = os.remove, args = (os.path.join(FID_FAKE_DIR, file),))
		t.start()
		threads.append(t)

	for t in threads:
		t.join()

	nb = 0
	threads = []

	def save_images(images: npt.NDArray[np.float32], i: int) -> None:

		for j in range(images.shape[0]):

			image = Image.fromarray(images[j])
			image.save(os.path.join(FID_FAKE_DIR, f'{str(i + j).zfill(math.ceil(math.log10(FID_NB_SAMPLES)))}.png'))

	while nb < FID_NB_SAMPLES:

		images = generator.generate(min(FID_NB_SAMPLES - nb, TEST_BATCH_SIZE))

		t = threading.Thread(target = save_images, args = (images, nb))
		t.start()
		threads.append(t)
		nb += images.shape[0]

	for t in threads:
		t.join()

	original_stdout = sys.stdout
	sys.stdout = open(os.devnull, 'w')

	fid = calculate_fid_given_paths([FID_REAL_DIR, FID_FAKE_DIR], FID_BATCH_SIZE, DEVICE, FID_DIMS, NB_WORKERS)

	sys.stdout.close()
	sys.stdout = original_stdout

	return fid
