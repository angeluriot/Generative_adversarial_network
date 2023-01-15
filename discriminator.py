import math
from tensorflow.keras import Model
from tensorflow.keras.layers import *

from settings import *
from layers import *


# Pixel space to feature space
def from_rgb(input, filters):

	model = EqualizedConv2D(filters, 1)(input)
	model = LeakyReLU(ALPHA)(model)

	return model


# Downsample
def downsample(input):

	# TODO ? Use StyleGAN3 non-sticking downsampling
	return AveragePooling2D()(input)


# Build a block
def build_block(input, filters):

	residual = EqualizedConv2D(filters, 1, use_bias = False)(input)
	residual = downsample(residual)

	model = EqualizedConv2D(filters, KERNEL_SIZE)(input)
	model = LeakyReLU(ALPHA)(model)

	model = EqualizedConv2D(filters, KERNEL_SIZE)(model)
	model = LeakyReLU(ALPHA)(model)

	model = downsample(model)
	model = Add()([model, residual])
	model = Lambda(lambda x: x / math.sqrt(2.))(model)

	return model


# Build the discriminator
def build_model():

	filters = get_filters(DIS_MIN_FILTERS, DIS_MAX_FILTERS, False)

	model_input = Input(shape = (IMAGE_SIZE, IMAGE_SIZE, NB_CHANNELS))
	model = from_rgb(model_input, filters[0])

	for i in range(NB_BLOCKS - 1):
		model = build_block(model, filters[i])

	model = MinibatchStdDev()(model)
	model = EqualizedConv2D(filters[-1], KERNEL_SIZE)(model)
	model = LeakyReLU(ALPHA)(model)

	model = Flatten()(model)
	model = EqualizedDense(filters[-1])(model)
	model = LeakyReLU(ALPHA)(model)

	model_output = EqualizedDense(1)(model)

	return Model(model_input, model_output)

