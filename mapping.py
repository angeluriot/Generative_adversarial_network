from tensorflow.keras import Model
from tensorflow.keras.layers import *

from settings import *
from layers import *


# Build the mapping network
def build_model():

	model_input = Input(shape = (LATENT_DIM,))
	model = PixelNorm()(model_input)

	for _ in range(MAPPING_LAYERS):
		model = EqualizedDense(LATENT_DIM, lr_multiplier = MAPPING_LR_RATIO)(model)
		model = LeakyReLU(ALPHA)(model)

	return Model(model_input, model)
