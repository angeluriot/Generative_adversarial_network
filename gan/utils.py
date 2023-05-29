import pickle
import numpy.typing as npt
import numpy as np
import datetime as dt
import tensorflow as tf
from keras import backend
from keras.optimizers import Optimizer

from gan.settings import *


# Reset the random seed
def reset_rand():

	now = dt.datetime.now()
	seconds_since_midnight = int((now - now.replace(hour = 0, minute = 0, second = 0, microsecond = 0)).total_seconds())
	np.random.seed(seconds_since_midnight)
	tf.random.set_seed(seconds_since_midnight)


# Normalize the image (numpy)
def norm_img(img):

	return np.clip((img.astype(np.float32) / 127.5) - 1.0, -1.0, 1.0)


# Normalize the image (tensorflow)
def tf_norm_img(img):

	return tf.clip_by_value((tf.cast(img, tf.float32) / 127.5) - 1.0, -1.0, 1.0)


# Denormalize the image (numpy)
def denorm_img(img):

	return ((np.clip(img, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)


# Denormalize the image (tensorflow)
def tf_denorm_img(img):

	return tf.cast((tf.clip_by_value(img, -1.0, 1.0) + 1.0) * 127.5, tf.uint8)


# Save optimizer state
def save_state(optimizer: Optimizer, path: str) -> None:

	variables = optimizer.variables()
	weights = [backend.get_value(var) for var in variables]
	pickle.dump(weights, open(path, 'wb'))


# Load optimizer state
def load_state(optimizer: Optimizer, path: str) -> None:

	variables = optimizer.variables()
	weights = pickle.load(open(path, 'rb'))

	for var, weight in zip(variables, weights):
		backend.set_value(var, weight)


# Generate z
def gen_z(nb: int) -> npt.NDArray[np.float32]:
	return np.random.normal(size = (nb, LATENT_DIM))


# Generate noise
def gen_noise(nb: int) -> list[npt.NDArray[np.float32]]:
	return [np.random.normal(0.0, 1.0, (nb, IMAGE_SIZE, IMAGE_SIZE, 1)) for _ in range((NB_BLOCKS * 2) - 1)]
