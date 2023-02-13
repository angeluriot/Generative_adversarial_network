import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model

from settings import *
import utils
import mapping, generator, discriminator
from augmentation import *


# The class for the GAN
class GAN(Model):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)

		self.mapping = mapping.build_model()
		self.generator = generator.build_model()
		self.discriminator = discriminator.build_model()

		self.ma_mapping = clone_model(self.mapping)
		self.ma_mapping.set_weights(self.mapping.get_weights())

		self.ma_generator = clone_model(self.generator)
		self.ma_generator.set_weights(self.generator.get_weights())

		self.step = 0
		self.tf_step = tf.Variable(self.step, dtype = tf.int32, trainable = False, name = "step")


	# Compile the model
	def compile(self, **kwargs):

		super().compile(**kwargs)
		self.generator_optimizer = Adam(LEARNING_RATE, BETA_1, BETA_2, EPSILON)
		self.discriminator_optimizer = Adam(LEARNING_RATE, BETA_1, BETA_2, EPSILON)


	# Print a summary of the model
	def summary(self, line_length = None, positions = None, print_fn = None):

		print("Mapping: {} -> {} | {:,} parameters".format(self.mapping.input_shape, self.mapping.output_shape, self.mapping.count_params()))

		print("Generator: [{}, {} * {}, {} * {}] -> {} | {:,} parameters".format(
			self.generator.input_shape[0],
			self.generator.input_shape[1], NB_BLOCKS,
			self.generator.input_shape[-1], (NB_BLOCKS * 2) - 1,
			self.generator.output_shape,
			self.generator.count_params()))

		print("Discriminator: {} -> {} | {:,} parameters".format(self.discriminator.input_shape, self.discriminator.output_shape, self.discriminator.count_params()))


	# Save the model
	def save_weights(self, dir):

		path = os.path.join(dir, "model_" + str(self.step // SAVE_FREQUENCY))

		if not os.path.exists(path):
			os.makedirs(path)

		self.mapping.save_weights(os.path.join(path, "mapping.h5"))
		self.generator.save_weights(os.path.join(path, "generator.h5"))
		self.discriminator.save_weights(os.path.join(path, "discriminator.h5"))
		self.ma_mapping.save_weights(os.path.join(path, "ma_mapping.h5"))
		self.ma_generator.save_weights(os.path.join(path, "ma_generator.h5"))


	# Load the model
	def load_weights(self, dir):

		folder = ""
		i = 0

		while True:

			if os.path.exists(os.path.join(dir, "model_" + str(i))):
				folder = os.path.join(dir, "model_" + str(i))

			else:
				break

			i += 1

		if folder != "":

			self.mapping.load_weights(os.path.join(folder, "mapping.h5"))
			self.generator.load_weights(os.path.join(folder, "generator.h5"))
			self.discriminator.load_weights(os.path.join(folder, "discriminator.h5"))
			self.ma_mapping.load_weights(os.path.join(folder, "ma_mapping.h5"))
			self.ma_generator.load_weights(os.path.join(folder, "ma_generator.h5"))

			self.step = (i - 1) * SAVE_FREQUENCY
			self.tf_step.assign(self.step)

		return folder != ""


	# Apply moving average
	def moving_average(self):

		for i in range(len(self.mapping.layers)):

			weights = self.mapping.layers[i].get_weights()
			old_weights = self.ma_mapping.layers[i].get_weights()
			new_weights = []

			for j in range(len(weights)):
				new_weights.append(old_weights[j] * MA_BETA + (1. - MA_BETA) * weights[j])

			self.ma_mapping.layers[i].set_weights(new_weights)

		for i in range(len(self.generator.layers)):

			weights = self.generator.layers[i].get_weights()
			old_weights = self.ma_generator.layers[i].get_weights()
			new_weights = []

			for j in range(len(weights)):
				new_weights.append(old_weights[j] * MA_BETA + (1. - MA_BETA) * weights[j])

			self.ma_generator.layers[i].set_weights(new_weights)


	# Give the output of the model from the inputs
	def predict(self, z, noise):

		generations = np.zeros((z.shape[0], IMAGE_SIZE, IMAGE_SIZE, NB_CHANNELS), dtype = np.uint8)

		for i in range(0, z.shape[0], BATCH_SIZE):

			size = min(BATCH_SIZE, z.shape[0] - i)
			const_input = [tf.ones((size, 1))]
			w = tf.convert_to_tensor(self.ma_mapping(z[i:i + size]))
			n = [tf.convert_to_tensor(j[i:i + size]) for j in noise]
			gen = self.ma_generator(const_input + ([w] * NB_BLOCKS) + n)
			generations[i:i + size, :, :, :] = utils.denorm_img(gen.numpy())

		return generations


	# Generate images
	def generate(self, nb):

		z = np.random.normal(0., 1., (nb, LATENT_DIM))
		noise = np.random.normal(0., 1., ((NB_BLOCKS * 2) - 1, nb, IMAGE_SIZE, IMAGE_SIZE, 1))

		return self.predict(z, noise)


	# Give W vectors for training
	def get_w(self, batch_size):

		rand = tf.random.uniform(shape = (), minval = 0., maxval = 1., dtype = tf.float32)

		# Style mixing
		if rand < STYLE_MIX_PROBA:

			cross_over_point = tf.random.uniform(shape = (), minval = 1, maxval = NB_BLOCKS, dtype = tf.int32)

			z1 = tf.random.normal(shape = (batch_size, LATENT_DIM))
			z2 = tf.random.normal(shape = (batch_size, LATENT_DIM))

			w1 = self.mapping(z1, training = True)
			w2 = self.mapping(z2, training = True)
			w = []

			for i in range(NB_BLOCKS):

				if i < cross_over_point:
					w_i = w1

				else:
					w_i = w2

				w.append(w_i)

			return w

		# No style mixing
		else:

			z = tf.random.normal(shape = (batch_size, LATENT_DIM))
			w = self.mapping(z, training = True)

			return [w] * NB_BLOCKS


	# Give noise for training
	def get_noise(self, batch_size):

		return [tf.random.normal((batch_size, IMAGE_SIZE, IMAGE_SIZE, 1)) for _ in range((NB_BLOCKS * 2) - 1)]


	# Compute the loss of the generator
	def generator_loss(self, fake_output):

		return tf.reduce_mean(tf.nn.softplus(-fake_output))


	# Compute the loss of the discriminator
	def discriminator_loss(self, real_output, fake_output):

		return tf.reduce_mean(tf.nn.softplus(-real_output)) + tf.reduce_mean(tf.nn.softplus(fake_output))


	# Compute the gradient penalty
	def gradient_penalty(self, real_output, real_images):

		gradients = tf.gradients(tf.reduce_sum(real_output), [real_images])[0]
		gradient_penalty = tf.reduce_sum(tf.square(gradients), axis = [1, 2, 3])

		return tf.reduce_mean(gradient_penalty) * GRADIENT_PENALTY_COEF * 0.5 * GRADIENT_PENALTY_INTERVAL


	# Train step
	@tf.function
	def train_step(self, data):

		batch_size = tf.shape(data)[0]
		const_input = [tf.ones((batch_size, 1))]
		noise = self.get_noise(batch_size)
		gradient_penalty = 0.

		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

			# Generate images
			w = self.get_w(batch_size)
			fake_images = self.generator(const_input + w + noise, training = True)

			# Data augmentation
			data = data_augmentation(data)
			fake_images = data_augmentation(fake_images)

			# Get discriminator outputs
			real_output = self.discriminator(data, training = True)
			fake_output = self.discriminator(fake_images, training = True)

			# Compute losses
			gen_loss = self.generator_loss(fake_output)
			disc_loss = self.discriminator_loss(real_output, fake_output)

			# Compute gradient penalty
			if self.tf_step % GRADIENT_PENALTY_INTERVAL == 0:
				gradient_penalty = self.gradient_penalty(real_output, data)

			# Get gradients
			generator_weights = (self.mapping.trainable_weights + self.generator.trainable_weights)
			generator_grad = gen_tape.gradient(gen_loss, generator_weights)
			discriminator_grad = disc_tape.gradient(disc_loss + gradient_penalty, self.discriminator.trainable_variables)

			# Update weights
			self.generator_optimizer.apply_gradients(zip(generator_grad, generator_weights))
			self.discriminator_optimizer.apply_gradients(zip(discriminator_grad, self.discriminator.trainable_variables))

			return {
				"Generator loss": gen_loss,
				"Discriminator loss": disc_loss,
				"Gradient penalty": gradient_penalty
			}
