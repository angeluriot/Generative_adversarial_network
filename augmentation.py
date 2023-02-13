import math
import tensorflow as tf
import tensorflow_addons as tfa

from settings import *


# Apply random modifications to the input data
def data_augmentation(data):

	batch_size = tf.shape(data)[0]

	if PIXEL_AUGMENTATION:

		# Horizontal flips
		if not FLIP_DATASET:
			rand_tensor = tf.repeat(tf.random.uniform((batch_size,), minval = 0., maxval = 1., dtype = tf.float32), IMAGE_SIZE * IMAGE_SIZE * NB_CHANNELS)
			rand_tensor = tf.reshape(rand_tensor, (batch_size, IMAGE_SIZE, IMAGE_SIZE, NB_CHANNELS))
			flipped_data = tf.image.random_flip_left_right(data)
			data = tf.where(rand_tensor < AUGMENTATION_PROBA, flipped_data, data)

		# 90° rotations
		angles = tf.random.uniform((batch_size,), minval = 0, maxval = 4, dtype = tf.int32)
		rand_tensor = tf.random.uniform((batch_size,), minval = 0., maxval = 1., dtype = tf.float32)
		angles = tf.where(rand_tensor < AUGMENTATION_PROBA, angles, tf.zeros_like(angles))
		data = tfa.image.rotate(data, tf.cast(angles, tf.float32) * (math.pi / 2.), interpolation = 'nearest', fill_mode = 'nearest')

		# Integer translations
		translations = tf.random.uniform((batch_size, 2), minval = int(-0.125 * IMAGE_SIZE), maxval = int(0.125 * IMAGE_SIZE), dtype = tf.int32)
		rand_tensor = tf.reshape(tf.repeat(tf.random.uniform((batch_size,), minval = 0., maxval = 1., dtype = tf.float32), 2), (batch_size, 2))
		translations = tf.where(rand_tensor < AUGMENTATION_PROBA, translations, tf.zeros_like(translations))
		data = tfa.image.translate(data, tf.cast(translations, tf.float32), interpolation = 'nearest', fill_mode = 'reflect')

	if GEOMETRIC_AUGMENTATION:

		# Isotropic scaling
		scales = tf.random.uniform((batch_size,), minval = 1. - 0.125, maxval = 1. + 0.125, dtype = tf.float32)
		rand_tensor = tf.random.uniform((batch_size,), minval = 0., maxval = 1., dtype = tf.float32)
		scales = tf.where(rand_tensor < AUGMENTATION_PROBA, scales, tf.ones_like(scales))
		zeros = tf.zeros((batch_size,), dtype = tf.float32)
		transform = [scales, zeros, ((1. - scales) * IMAGE_SIZE) / 2., zeros, scales, ((1. - scales) * IMAGE_SIZE) / 2., zeros, zeros]
		transform = tf.transpose(transform, [1, 0])
		data = tfa.image.transform(data, transform, interpolation = 'bilinear', fill_mode = 'reflect')

		# Arbitrary rotations
		angles = tf.random.uniform((batch_size,), minval = 0., maxval = math.pi * 2., dtype = tf.float32)
		rand_tensor = tf.random.uniform((batch_size,), minval = 0., maxval = 1., dtype = tf.float32)
		angles = tf.where(rand_tensor < AUGMENTATION_PROBA, angles, tf.zeros_like(angles))
		data = tfa.image.rotate(data, angles, interpolation = 'bilinear', fill_mode = 'reflect')

		# Anisotropic scaling
		x_scales = tf.random.uniform((batch_size,), minval = 1. - 0.125, maxval = 1. + 0.125, dtype = tf.float32)
		y_scales = tf.random.uniform((batch_size,), minval = 1. - 0.125, maxval = 1. + 0.125, dtype = tf.float32)
		rand_tensor = tf.random.uniform((batch_size,), minval = 0., maxval = 1., dtype = tf.float32)
		x_scales = tf.where(rand_tensor < AUGMENTATION_PROBA, x_scales, tf.ones_like(x_scales))
		y_scales = tf.where(rand_tensor < AUGMENTATION_PROBA, y_scales, tf.ones_like(y_scales))
		zeros = tf.zeros((batch_size,), dtype = tf.float32)
		transform = [x_scales, zeros, ((1. - x_scales) * IMAGE_SIZE) / 2., zeros, y_scales, ((1. - y_scales) * IMAGE_SIZE) / 2., zeros, zeros]
		transform = tf.transpose(transform, [1, 0])
		data = tfa.image.transform(data, transform, interpolation = 'bilinear', fill_mode = 'reflect')

		# Fractional translations
		translations = tf.random.uniform((batch_size, 2), minval = -0.125 * IMAGE_SIZE, maxval = 0.125 * IMAGE_SIZE, dtype = tf.float32)
		rand_tensor = tf.reshape(tf.repeat(tf.random.uniform((batch_size,), minval = 0., maxval = 1., dtype = tf.float32), 2), (batch_size, 2))
		translations = tf.where(rand_tensor < AUGMENTATION_PROBA, translations, tf.zeros_like(translations))
		data = tfa.image.translate(data, translations, interpolation = 'bilinear', fill_mode = 'reflect')

	return data
