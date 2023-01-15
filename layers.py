import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import *

from settings import *


# Give the list of filters for each layer
def get_filters(min_filters, max_filters, max_first):

	filters = []

	for i in range(NB_BLOCKS):
		filters.append(min(int(min_filters * (2 ** i)), max_filters))

	if max_first:
		return filters[::-1]

	return filters


# Normalize the input latent vector
class PixelNorm(Layer):

	def __init__(self, epsilon = 1e-8, **kwargs):

		super().__init__(**kwargs)
		self.epsilon = epsilon


	def get_config(self):

		config = super().get_config().copy()

		config.update({
			'epsilon': self.epsilon
		})

		return config


	def call(self, inputs):

		return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis = -1, keepdims = True) + self.epsilon)


# Equalized learning rate dense layer
class EqualizedDense(Layer):

	def __init__(self, units, bias_init = 0., use_bias = True, gain = GAIN, lr_multiplier = 1., **kwargs):

		super().__init__(**kwargs)
		self.units = units
		self.bias_init = bias_init
		self.use_bias = use_bias
		self.gain = gain
		self.lr_multiplier = lr_multiplier


	def get_config(self):

		config = super().get_config().copy()

		config.update({
			'units': self.units,
			'bias_init': self.bias_init,
			'use_bias': self.use_bias,
			'gain': self.gain,
			'lr_multiplier': self.lr_multiplier
		})

		return config


	def build(self, input_shape):

		super().build(input_shape)

		self.weight = self.add_weight(
			shape = (input_shape[-1], self.units),
			initializer = RandomNormal(mean = 0., stddev = 1. / self.lr_multiplier),
			trainable = True,
			name = "kernel"
		)

		if self.use_bias:
			self.bias = self.add_weight(
				shape = (self.units,),
				initializer = Constant(self.bias_init / self.lr_multiplier),
				trainable = True,
				name = "bias"
			)

		fan_in = input_shape[-1]
		self.scale = self.gain / np.sqrt(fan_in)


	def call(self, inputs):

		output = tf.matmul(inputs, self.scale * self.weight * self.lr_multiplier)

		if self.use_bias:
			return output + (self.bias * self.lr_multiplier)

		return output


# Equalized learning rate convolutional layer
class EqualizedConv2D(Layer):

	def __init__(self, filters, kernel_size, bias_init = 0., use_bias = True, gain = GAIN, lr_multiplier = 1., **kwargs):

		super().__init__(**kwargs)
		self.filters = filters
		self.kernel_size = kernel_size
		self.bias_init = bias_init
		self.use_bias = use_bias
		self.gain = gain
		self.lr_multiplier = lr_multiplier


	def get_config(self):

		config = super().get_config().copy()

		config.update({
			'filters': self.filters,
			'kernel_size': self.kernel_size,
			'bias_init': self.bias_init,
			'use_bias': self.use_bias,
			'gain': self.gain,
			'lr_multiplier': self.lr_multiplier
		})

		return config


	def build(self, input_shape):

		super().build(input_shape)

		self.kernel = self.add_weight(
			shape = (self.kernel_size, self.kernel_size, input_shape[-1], self.filters),
			initializer = RandomNormal(mean = 0., stddev = 1. / self.lr_multiplier),
			trainable = True,
			name = "kernel",
		)

		if self.use_bias:

			self.bias = self.add_weight(
				shape = (self.filters,),
				initializer = Constant(self.bias_init / self.lr_multiplier),
				trainable = True,
				name = "bias"
			)

		fan_in = self.kernel_size * self.kernel_size * input_shape[-1]
		self.scale = self.gain / np.sqrt(fan_in)


	def call(self, inputs):

		output = tf.nn.conv2d(inputs, self.scale * self.kernel * self.lr_multiplier, strides = 1, padding = "SAME", data_format = "NHWC")

		if self.use_bias:
			return output + (self.bias * self.lr_multiplier)[None, None, None, :]

		return output


# Add a trainable amount of noise to the input
class AddNoise(Layer):

	def __init__(self, lr_multiplier = 1., **kwargs):

		super().__init__(**kwargs)
		self.lr_multiplier = lr_multiplier


	def get_config(self):

		config = super().get_config().copy()

		config.update({
			'lr_multiplier': self.lr_multiplier
		})

		return config


	def build(self, input_shape):

		super().build(input_shape)

		self.noise_scale = self.add_weight(
			shape = None,
			initializer = Constant(0. / self.lr_multiplier),
			trainable = True,
			name = "noise_scale",
		)


	def call(self, inputs):

		return inputs[0] + (self.noise_scale * self.lr_multiplier * inputs[1])


# Add a trainable bias to the input
class Bias(Layer):

	def __init__(self, bias_init = 0., lr_multiplier = 1., **kwargs):

		super().__init__(**kwargs)
		self.bias_init = bias_init
		self.lr_multiplier = lr_multiplier


	def get_config(self):

		config = super().get_config().copy()

		config.update({
			'bias_init': self.bias_init,
			'lr_multiplier': self.lr_multiplier
		})

		return config


	def build(self, input_shape):

		super().build(input_shape)

		self.bias = self.add_weight(
			shape = (input_shape[-1],),
			initializer = Constant(self.bias_init / self.lr_multiplier),
			trainable = True,
			name = "bias"
		)


	def call(self, inputs):

		return inputs + (self.bias * self.lr_multiplier)[None, None, None, :]


# Equalized learning rate modulated convolutional layer
class ModulatedConv2D(Layer):

	def __init__(self, filters, kernel_size, demodulate = True, epsilon = 1e-8, gain = GAIN, lr_multiplier = 1., **kwargs):

		super().__init__(**kwargs)
		self.filters = filters
		self.kernel_size = kernel_size
		self.demodulate = demodulate
		self.epsilon = epsilon
		self.gain = gain
		self.lr_multiplier = lr_multiplier


	def get_config(self):

		config = super().get_config().copy()

		config.update({
			'filters': self.filters,
			'kernel_size': self.kernel_size,
			'demodulate': self.demodulate,
			'epsilon': self.epsilon,
			'gain': self.gain,
			'lr_multiplier': self.lr_multiplier
		})

		return config


	def build(self, input_shape):

		super().build(input_shape)

		x_shape = input_shape[0]

		self.kernel = self.add_weight(
			shape = (self.kernel_size, self.kernel_size, x_shape[-1], self.filters),
			initializer = RandomNormal(mean = 0., stddev = 1. / self.lr_multiplier),
			trainable = True,
			name = "kernel",
		)

		fan_in = self.kernel_size * self.kernel_size * x_shape[-1]
		self.scale = self.gain / np.sqrt(fan_in)


	def call(self, inputs):

		x = inputs[0]
		style = inputs[1]

		x = tf.transpose(x, [0, 3, 1, 2])

		# Modulate
		ww = (self.scale * self.kernel * self.lr_multiplier)[None, :, :, :, :]
		ww *= style[:, None, None, :, None]

		# Demodulate
		if self.demodulate:
			sigma = tf.math.rsqrt(tf.reduce_sum(tf.square(ww), axis = [1, 2, 3]) + self.epsilon)
			ww *= sigma[:, None, None, None, :]

		# Reshape input
		x = tf.reshape(x, (1, -1, x.shape[2], x.shape[3]))
		w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), (ww.shape[1], ww.shape[2], ww.shape[3], -1))

		# Convolution
		output = tf.nn.conv2d(x, w, strides = 1, padding = "SAME", data_format = "NCHW")

		# Reshape output
		output = tf.reshape(output, (-1, self.filters, x.shape[2], x.shape[3]))

		return tf.transpose(output, [0, 2, 3, 1])


# Add minibatch standard deviation to the input
class MinibatchStdDev(Layer):

	def __init__(self, epsilon = 1e-8, **kwargs):

		super().__init__(**kwargs)
		self.epsilon = epsilon


	def get_config(self):

		config = super().get_config().copy()

		config.update({
			'epsilon': self.epsilon
		})

		return config


	def call(self, inputs):

		mean = tf.reduce_mean(inputs, axis = 0, keepdims = True)
		squ_diffs = tf.square(inputs - mean)
		mean_sq_diff = tf.reduce_mean(squ_diffs, axis = 0, keepdims = True) + self.epsilon
		stdev = tf.sqrt(mean_sq_diff)
		mean_pix = tf.reduce_mean(stdev, keepdims = True)
		shape = tf.shape(inputs)
		output = tf.tile(mean_pix, (shape[0], shape[1], shape[2], 1))

		return tf.concat([inputs, output], axis = -1)
