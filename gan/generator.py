from keras.engine.keras_tensor import KerasTensor
from keras import Model
from keras.layers import *

from gan.settings import *
from gan.layers import *


# Feature space to pixel space
def to_rgb(input: KerasTensor, w: KerasTensor) -> KerasTensor:

	style = EqualizedDense(input.shape[-1], bias_init = 1.0)(w)
	model = ModulatedConv2D(NB_CHANNELS, 1, False)([input, style])
	model = Bias()(model)

	return model


# Upsample
def upsample(input: KerasTensor) -> KerasTensor:

	# TODO ? Use StyleGAN3 non-sticking upsampling
	return UpSampling2D(interpolation = 'bilinear')(input)


# Build style block
def build_style_block(input: KerasTensor, w: KerasTensor, noise: KerasTensor, filters: int) -> KerasTensor:

	style = EqualizedDense(input.shape[-1], bias_init = 1.0)(w)
	model = ModulatedConv2D(filters, KERNEL_SIZE, True)([input, style])

	n = Cropping2D((noise.shape[1] - model.shape[1]) // 2)(noise)
	model = AddNoise()([model, n])
	model = Bias()(model)
	model = LeakyReLU(ALPHA)(model)

	return model


# Build a block
def build_block(input: KerasTensor, w: KerasTensor, noise_1: KerasTensor, noise_2: KerasTensor, filters: int) -> KerasTensor:

	model = build_style_block(input, w, noise_1, filters)
	model = build_style_block(model, w, noise_2, filters)

	rgb = to_rgb(model, w)

	return model, rgb


# Build the generator
def build_model() -> Model:

	filters = get_filters(GEN_MIN_FILTERS, GEN_MAX_FILTERS, True)

	model_input = Input(shape = (1,))
	w_inputs = [Input(shape = (LATENT_DIM,)) for _ in range(NB_BLOCKS)]
	noise_inputs = [Input(shape = (IMAGE_SIZE, IMAGE_SIZE, 1)) for _ in range((NB_BLOCKS * 2) - 1)]

	model = EqualizedDense(MIN_IMAGE_SIZE * MIN_IMAGE_SIZE * filters[0], use_bias = False)(model_input)
	model = Reshape((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, filters[0]))(model)

	model = build_style_block(model, w_inputs[0], noise_inputs[0], filters[0])
	rgb = to_rgb(model, w_inputs[0])

	for i in range(1, NB_BLOCKS):
		model = upsample(model)
		model, new_rgb = build_block(model, w_inputs[i], noise_inputs[(i * 2) - 1], noise_inputs[i * 2], filters[i])
		rgb = Add()([upsample(rgb), new_rgb])

	model_output = Activation('tanh')(rgb)

	model = Model([model_input] + w_inputs + noise_inputs, model_output)

	for i in range(len(model.weights)):
		model.weights[i]._handle_name = model.weights[i].name + "_gen_" + str(i)

	return model
