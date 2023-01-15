import os
import numpy as np
from PIL import Image
from tensorflow.keras.callbacks import Callback

from settings import *


# Update variables and apply moving average
class Updates(Callback):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)


	def on_batch_begin(self, batch, logs = None):

		self.model.step += 1
		self.model.tf_step.assign(self.model.step)


	def on_batch_end(self, batch, logs = None):

		self.model.moving_average()


# Save samples
class SaveSamples(Callback):

	def __init__(self, z, noise, **kwargs):

		super().__init__(**kwargs)
		self.z = z
		self.noise = noise
		self.epoch = 0


	def on_batch_end(self, batch, logs = None):

		if self.model.step % SAVE_FREQUENCY == 0 or self.model.step == 1:

			generations = self.model.predict(self.z, list(self.noise))

			output_image = np.full((
				MARGIN + (OUTPUT_SHAPE[1] * (generations.shape[2] + MARGIN)),
				MARGIN + (OUTPUT_SHAPE[0] * (generations.shape[1] + MARGIN)),
				generations.shape[3]), 255, dtype = np.uint8
			)

			i = 0

			for row in range(OUTPUT_SHAPE[1]):
				for col in range(OUTPUT_SHAPE[0]):
					r = row * (generations.shape[2] + MARGIN) + MARGIN
					c = col * (generations.shape[1] + MARGIN) + MARGIN
					output_image[r:r + generations.shape[2], c:c + generations.shape[1]] = generations[i]
					i += 1

			if not os.path.exists(SAMPLES_DIR):
				os.makedirs(SAMPLES_DIR)

			img = Image.fromarray(output_image)
			img.save(os.path.join(SAMPLES_DIR, "image_" + str(self.model.step // SAVE_FREQUENCY) + ".png"))
			img.save(os.path.join(OUTPUT_DIR, "last_image.png"))


	def on_epoch_begin(self, epoch, logs = None):

		self.epoch = epoch


# Save models
class SaveModels(Callback):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)


	def on_batch_end(self, batch, logs = None):

		if self.model.step % SAVE_FREQUENCY == 0 or self.model.step == 1:
			self.model.save_weights(MODELS_DIR)
