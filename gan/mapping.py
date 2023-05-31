from keras import Model
from keras.layers import *

from gan.settings import *
from gan.layers import *


# Build the mapping network
def build_model() -> Model:

	model_input = Input(shape = (LATENT_DIM,))
	model = PixelNorm()(model_input)

	for _ in range(MAPPING_LAYERS):
		model = EqualizedDense(LATENT_DIM, lr_multiplier = MAPPING_LR_RATIO)(model)
		model = LeakyReLU(ALPHA)(model)

	model = Model(model_input, model)

	for i in range(len(model.weights)):
		model.weights[i]._handle_name = model.weights[i].name + "_map_" + str(i)

	return model
