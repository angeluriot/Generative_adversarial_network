import os, pickle
import numpy.typing as npt
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras import Model
from keras.optimizers import Adam
from keras.models import clone_model

from gan.settings import *
from gan.layers import *
from gan import utils
from gan import mapping, generator, discriminator
from gan.augmentation import *


# The class for the GAN
class GAN(Model):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)

		self.mapping: Model = mapping.build_model()
		self.generator: Model = generator.build_model()
		self.discriminator: Model = discriminator.build_model()

		self.ma_mapping: Model = clone_model(self.mapping)
		self.ma_mapping.set_weights(self.mapping.get_weights())

		self.ma_generator: Model = clone_model(self.generator)
		self.ma_generator.set_weights(self.generator.get_weights())

		if ACCUMULATION_STEPS > 1:

			self.generator_gradients: list[tf.Variable] = []

			for i in range(len(self.mapping.trainable_variables)):
				self.generator_gradients.append(tf.Variable(tf.zeros_like(self.mapping.trainable_variables[i]), trainable = False))

			for i in range(len(self.generator.trainable_variables)):
				self.generator_gradients.append(tf.Variable(tf.zeros_like(self.generator.trainable_variables[i]), trainable = False))

			self.discriminator_gradients: list[tf.Variable] = []

			for i in range(len(self.discriminator.trainable_variables)):
				self.discriminator_gradients.append(tf.Variable(tf.zeros_like(self.discriminator.trainable_variables[i]), trainable = False))

			self.accumulated_steps: tf.Variable = tf.Variable(0, dtype = tf.int32, trainable = False, name = 'accumulated_steps')

		self.step: int = 0
		self.tf_step: tf.Variable = tf.Variable(self.step, dtype = tf.int32, trainable = False, name = 'step')

		self.mean_w: npt.NDArray[np.float32] = np.zeros((LATENT_DIM,), dtype = np.float32)


	# Compile the model
	def compile(self, **kwargs) -> None:

		super().compile(**kwargs)
		self.generator_optimizer = Adam(LEARNING_RATE, BETA_1, BETA_2, EPSILON)
		self.discriminator_optimizer = Adam(LEARNING_RATE, BETA_1, BETA_2, EPSILON)


	# Print a summary of the model
	def summary(self, line_length = None, positions = None, print_fn = None) -> None:

		print('Mapping: {} -> {} | {:,} parameters'.format(self.mapping.input_shape, self.mapping.output_shape, self.mapping.count_params()))

		print('Generator: [{}, {} * {}, {} * {}] -> {} | {:,} parameters'.format(
			self.generator.input_shape[0],
			self.generator.input_shape[1], NB_BLOCKS,
			self.generator.input_shape[-1], (NB_BLOCKS * 2) - 1,
			self.generator.output_shape,
			self.generator.count_params()
		))

		print('Discriminator: {} -> {} | {:,} parameters'.format(self.discriminator.input_shape, self.discriminator.output_shape, self.discriminator.count_params()))


	# Save the model
	def save(self, path: str) -> None:

		if not os.path.exists(path):
			os.makedirs(path)

		self.mapping.save_weights(os.path.join(path, 'mapping.h5'))
		self.generator.save_weights(os.path.join(path, 'generator.h5'))
		self.discriminator.save_weights(os.path.join(path, 'discriminator.h5'))
		self.ma_mapping.save_weights(os.path.join(path, 'ma_mapping.h5'))
		self.ma_generator.save_weights(os.path.join(path, 'ma_generator.h5'))

		utils.save_state(self.generator_optimizer, os.path.join(path, 'generator_optimizer.pkl'))
		utils.save_state(self.discriminator_optimizer, os.path.join(path, 'discriminator_optimizer.pkl'))

		pickle.dump(self.step, open(os.path.join(path, 'step.pkl'), 'wb'))


	# Load the model
	def load(self, path: str) -> bool:

		if os.path.exists(os.path.join(path, 'mapping.h5')):
			self.mapping.load_weights(os.path.join(path, 'mapping.h5'))

		if os.path.exists(os.path.join(path, 'generator.h5')):
			self.generator.load_weights(os.path.join(path, 'generator.h5'))

		if os.path.exists(os.path.join(path, 'discriminator.h5')):
			self.discriminator.load_weights(os.path.join(path, 'discriminator.h5'))

		if os.path.exists(os.path.join(path, 'ma_mapping.h5')):
			self.ma_mapping.load_weights(os.path.join(path, 'ma_mapping.h5'))

		if os.path.exists(os.path.join(path, 'ma_generator.h5')):
			self.ma_generator.load_weights(os.path.join(path, 'ma_generator.h5'))

		if os.path.exists(os.path.join(path, 'generator_optimizer.pkl')):
			utils.load_state(self.generator_optimizer, os.path.join(path, 'generator_optimizer.pkl'))

		if os.path.exists(os.path.join(path, 'discriminator_optimizer.pkl')):
			utils.load_state(self.discriminator_optimizer, os.path.join(path, 'discriminator_optimizer.pkl'))

		if os.path.exists(os.path.join(path, 'step.pkl')):
			self.step = pickle.load(open(os.path.join(path, 'step.pkl'), 'rb'))
			self.tf_step.assign(self.step)

		return self.step > 0


	# Apply moving average
	def moving_average(self) -> None:

		for i in range(len(self.mapping.layers)):

			weights = self.mapping.layers[i].get_weights()
			old_weights = self.ma_mapping.layers[i].get_weights()
			new_weights = []

			for j in range(len(weights)):
				new_weights.append(old_weights[j] * MA_BETA + (1.0 - MA_BETA) * weights[j])

			self.ma_mapping.layers[i].set_weights(new_weights)

		for i in range(len(self.generator.layers)):

			weights = self.generator.layers[i].get_weights()
			old_weights = self.ma_generator.layers[i].get_weights()
			new_weights = []

			for j in range(len(weights)):
				new_weights.append(old_weights[j] * MA_BETA + (1.0 - MA_BETA) * weights[j])

			self.ma_generator.layers[i].set_weights(new_weights)


	# Find mean W latent vector
	def compute_mean_w(self, nb: int = 100_000) -> npt.NDArray[np.float32]:

		z = utils.gen_z(nb)
		w = self.z_to_w(z)

		self.mean_w = np.mean(w, axis = 0)

		return self.mean_w


	# Turn Z seed into W latent vector
	def z_to_w(self, z: npt.NDArray[np.float32], psi: float = 1.0) -> npt.NDArray[np.float32]:

		w = np.zeros((z.shape[0], LATENT_DIM), dtype = np.float32)

		for i in range(0, z.shape[0], BATCH_SIZE * 100):

			size = min(BATCH_SIZE * 100, z.shape[0] - i)
			w[i:i + size, :] = self.ma_mapping(tf.convert_to_tensor(z[i:i + size])).numpy()

		return self.mean_w + psi * (w - self.mean_w)


	# Turn W latent vector into the final image
	def w_to_image(self, w: npt.NDArray[np.float32], noise: npt.NDArray[np.float32] | None = None, psi: float = 1.0) -> npt.NDArray[np.uint8]:

		generations = np.zeros((w.shape[0], IMAGE_SIZE, IMAGE_SIZE, NB_CHANNELS), dtype = np.uint8)
		w = self.mean_w + psi * (w - self.mean_w)

		if noise is None:
			noise = utils.gen_noise(w.shape[0])

		for i in range(0, w.shape[0], BATCH_SIZE * 2):

			size = min(BATCH_SIZE * 2, w.shape[0] - i)
			const_input = [tf.ones((size, 1))]
			n = [tf.convert_to_tensor(j[i:i + size]) for j in noise]
			gen = self.ma_generator(const_input + ([w[i:i + size]] * NB_BLOCKS) + n)
			generations[i:i + size, :, :, :] = utils.denorm_img(gen.numpy())

		return generations


	# Give the output of the model from the inputs
	def predict(self, z: npt.NDArray[np.float32], noise: npt.NDArray[np.float32] | None = None, psi: float = 1.0) -> npt.NDArray[np.uint8]:

		return self.w_to_image(self.z_to_w(z, psi), noise)


	# Generate images
	def generate(self, nb: int, psi: float = 1.0) -> npt.NDArray[np.uint8]:

		return self.predict(utils.gen_z(nb), None, psi)


	# Give W vectors for training
	def get_w(self, batch_size: int) -> list[tf.Tensor]:

		rand = tf.random.uniform(shape = (), minval = 0.0, maxval = 1.0, dtype = tf.float32)

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
	def get_noise(self, batch_size: int) -> list[tf.Tensor]:

		return [tf.random.normal((batch_size, IMAGE_SIZE, IMAGE_SIZE, 1)) for _ in range((NB_BLOCKS * 2) - 1)]


	# Compute the loss of the generator
	def generator_loss(self, fake_output: tf.Tensor) -> tf.Tensor:

		return tf.reduce_mean(tf.nn.softplus(-fake_output))


	# Compute the loss of the discriminator
	def discriminator_loss(self, real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:

		return tf.reduce_mean(tf.nn.softplus(-real_output)) + tf.reduce_mean(tf.nn.softplus(fake_output))


	# Compute the gradient penalty
	def gradient_penalty(self, real_output: tf.Tensor, real_images: tf.Tensor) -> tf.Tensor:

		gradients = tf.gradients(tf.reduce_sum(real_output), [real_images])[0]
		gradient_penalty = tf.reduce_sum(tf.square(gradients), axis = [1, 2, 3])

		return tf.reduce_mean(gradient_penalty) * GRADIENT_PENALTY_COEF * 0.5 * GRADIENT_PENALTY_INTERVAL


	# Train step
	@tf.function
	def train_step(self, data: tf.Tensor) -> dict[str, float]:

		batch_size = tf.shape(data)[0]
		const_input = [tf.ones((batch_size, 1))]
		noise = self.get_noise(batch_size)
		gradient_penalty = 0.0

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

			# If gradient accumulation is enabled
			if ACCUMULATION_STEPS > 1:

				# Accumulate gradients
				for i in range(len(generator_grad)):
					self.generator_gradients[i].assign_add(generator_grad[i] / float(ACCUMULATION_STEPS))

				for i in range(len(discriminator_grad)):
					self.discriminator_gradients[i].assign_add(discriminator_grad[i] / float(ACCUMULATION_STEPS))

				# Increment counter
				self.accumulated_steps.assign_add(1)

				if self.accumulated_steps >= ACCUMULATION_STEPS:

					# Update accumulated gradients
					self.generator_optimizer.apply_gradients(zip(self.generator_gradients, generator_weights))
					self.discriminator_optimizer.apply_gradients(zip(self.discriminator_gradients, self.discriminator.trainable_variables))

					# Reset accumulated gradients
					for i in range(len(self.generator_gradients)):
						self.generator_gradients[i].assign(tf.zeros_like(self.generator_gradients[i]))

					for i in range(len(self.discriminator_gradients)):
						self.discriminator_gradients[i].assign(tf.zeros_like(self.discriminator_gradients[i]))

					# Reset counter
					self.accumulated_steps.assign(0)

			# If gradient accumulation is disabled
			else:
				# Update weights
				self.generator_optimizer.apply_gradients(zip(generator_grad, generator_weights))
				self.discriminator_optimizer.apply_gradients(zip(discriminator_grad, self.discriminator.trainable_variables))

		return {
			'Generator loss': gen_loss,
			'Discriminator loss': disc_loss,
			'Gradient penalty': gradient_penalty
		}
