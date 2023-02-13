import numpy as np
import datetime as dt
import tensorflow as tf


# Reset the random seed
def reset_rand():

	now = dt.datetime.now()
	seconds_since_midnight = int((now - now.replace(hour = 0, minute = 0, second = 0, microsecond = 0)).total_seconds())
	np.random.seed(seconds_since_midnight)
	tf.random.set_seed(seconds_since_midnight)


# Normalize the image (numpy)
def norm_img(img):

	return np.clip((img.astype(np.float32) / 127.5) - 1., -1., 1.)


# Normalize the image (tensorflow)
def tf_norm_img(img):

	return tf.clip_by_value((tf.cast(img, tf.float32) / 127.5) - 1., -1., 1.)


# Denormalize the image (numpy)
def denorm_img(img):

	return ((np.clip(img, -1., 1.) + 1.) * 127.5).astype(np.uint8)


# Denormalize the image (tensorflow)
def tf_denorm_img(img):

	return tf.cast((tf.clip_by_value(img, -1., 1.) + 1.) * 127.5, tf.uint8)
