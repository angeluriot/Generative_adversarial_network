import os, pickle, time
from PIL import Image
import torch
from torch import nn

from training.settings import *
from training import data, metrics
from training.generator import *
from training.discriminator import *
from training import losses


class Trainer():

	def __init__(self, dataset: data.Dataset, generator: Generator, discriminator: Discriminator):

		self.dataset = dataset

		self.generator = generator
		self.discriminator = discriminator

		self.gen_optimizer = torch.optim.Adam(
			self.generator.parameters(),
			lr = LEARNING_RATE,
			betas = (BETA_1, BETA_2),
			eps = EPSILON
		)

		self.disc_optimizer = torch.optim.Adam(
			self.discriminator.parameters(),
			lr = LEARNING_RATE,
			betas = (BETA_1, BETA_2),
			eps = EPSILON
		)

		self.sample_z = self.generator.gen_z(OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1])

		self.step = 0
		self.images_seen = 0
		self.epochs = 0.0
		self.augmentation_proba = utils.get_dict_value(AUGMENTATION_PROBAS, 0.0)
		self.fid = 0.0
		self.metrics = {
			'steps': [],
			'images': [],
			'epochs': [],
			'time': [],
			'fid': []
		}


	# Save the models
	def save_models(self, path: str | list[str]) -> None:

		if type(path) == str:
			path = [path]

		for p in path:

			if not os.path.exists(p):
				os.makedirs(p)

			torch.save(self.generator.state_dict(), os.path.join(p, 'generator.pt'))
			torch.save(self.discriminator.state_dict(), os.path.join(p, 'discriminator.pt'))
			torch.save(self.gen_optimizer.state_dict(), os.path.join(p, 'gen_optimizer.pt'))
			torch.save(self.disc_optimizer.state_dict(), os.path.join(p, 'disc_optimizer.pt'))
			pickle.dump(self.step, open(os.path.join(p, 'step.pkl'), 'wb'))
			pickle.dump(self.sample_z, open(os.path.join(OUTPUT_DIR, 'sample_z.pkl'), 'wb'))
			pickle.dump(self.metrics, open(os.path.join(OUTPUT_DIR, 'metrics.pkl'), 'wb'))


	# Load the models
	def load_models(self, path: str) -> None:

		self.generator.load_state_dict(torch.load(os.path.join(path, 'generator.pt'), map_location = DEVICE))
		self.discriminator.load_state_dict(torch.load(os.path.join(path, 'discriminator.pt'), map_location = DEVICE))
		self.gen_optimizer.load_state_dict(torch.load(os.path.join(path, 'gen_optimizer.pt'), map_location = DEVICE))
		self.disc_optimizer.load_state_dict(torch.load(os.path.join(path, 'disc_optimizer.pt'), map_location = DEVICE))
		self.step = pickle.load(open(os.path.join(path, 'step.pkl'), 'rb')) + 1
		self.sample_z = pickle.load(open(os.path.join(OUTPUT_DIR, 'sample_z.pkl'), 'rb'))
		self.metrics = pickle.load(open(os.path.join(OUTPUT_DIR, 'metrics.pkl'), 'rb'))
		self.fid = self.metrics['fid'][-1] if len(self.metrics['fid']) > 0 else 0.0


	# Find previous session
	def find_previous_session(self) -> None:

		if os.path.exists(os.path.join(OUTPUT_DIR, 'last_model')):
			self.load_models(os.path.join(OUTPUT_DIR, 'last_model'))


	# Save samples
	def save_samples(self, path: str | list[str]) -> None:

		if type(path) == str:
			path = [path]

		images = self.generator.z_to_images(self.sample_z)
		images = utils.create_grid(images, OUTPUT_SHAPE)
		images = Image.fromarray(images)

		for p in path:

			if not os.path.exists(os.path.dirname(p)):
				os.makedirs(os.path.dirname(p))

			images.save(p)


	# Print
	def print(self, gen_loss: float, disc_loss: float) -> None:

		print(f'Steps: {self.step:,} | Images: {self.images_seen:,} | Epochs: {self.epochs:.3f} | Augment proba: {self.augmentation_proba:.3f}   ||   ' + \
			f'Gen loss: {gen_loss:.3f}   ||   Disc loss: {disc_loss:.4f}   ||   FID: {self.fid:.1f}          ', end = '\r')


	# Train the models
	def train(self) -> None:

		last_time = time.time()

		self.generator.train()
		self.discriminator.train()

		metrics.clone_dataset()

		# Training loop
		while True:

			# Import data asynchronously
			real_images = self.dataset.next().to(DEVICE, non_blocking = True)

			# =============== TRAIN DISCRIMINATOR =============== #

			self.discriminator.zero_grad()

			# Forward pass
			z = self.generator.gen_z(BATCH_SIZE)
			fake_images = self.generator(z)
			fake_scores = self.discriminator(fake_images.detach())
			real_scores = self.discriminator(real_images)

			# Discriminator loss
			disc_loss = losses.disc_loss(fake_scores, real_scores)

			# Backward pass
			disc_loss.backward()

			# Update weights
			self.discriminator.clean_nan()
			self.disc_optimizer.step()

			# =============== TRAIN GENERATOR =============== #

			self.generator.zero_grad()

			# Forward pass
			fake_scores = self.discriminator(fake_images)

			# Generator loss
			gen_loss = losses.gen_loss(fake_scores)

			# Backward pass
			gen_loss.backward()

			# Update weights
			self.generator.clean_nan()
			self.gen_optimizer.step()

			# ======================================== #

			# Save models
			if self.step % MODEL_SAVE_FREQUENCY == 0:
				i = self.step // MODEL_SAVE_FREQUENCY
				self.save_models(os.path.join(MODELS_DIR, f'model_n-{i}_step-{self.step}'))

			if self.step % min(MODEL_SAVE_FREQUENCY, SAMPLE_SAVE_FREQUENCY) == 0:
				self.save_models(os.path.join(OUTPUT_DIR, 'last_model'))

			# Save samples
			if self.step % SAMPLE_SAVE_FREQUENCY == 0:
				i = self.step // SAMPLE_SAVE_FREQUENCY
				self.save_samples([os.path.join(SAMPLES_DIR, f'sample_n-{i}_step-{self.step}.png'), os.path.join(OUTPUT_DIR, 'last_sample.png')])

			# Compute metrics
			if self.step % METRICS_FREQUENCY == 0:

				self.fid = metrics.compute_fid(self.generator)

				self.metrics['steps'].append(self.step)
				self.metrics['images'].append(self.images_seen)
				self.metrics['epochs'].append(self.epochs)
				self.metrics['fid'].append(self.fid)

				time_elapsed = time.time() - last_time
				last_time = time.time()

				self.metrics['time'].append(time_elapsed if len(self.metrics['time']) == 0 else self.metrics['time'][-1] + time_elapsed)

			# Print
			self.print(gen_loss.item(), disc_loss.item())

			# Update step
			self.step += 1
			self.images_seen = self.step * BATCH_SIZE * ACCUMULATION_STEPS
			self.epochs = self.images_seen / self.dataset.size()
			self.augmentation_proba = utils.get_dict_value(AUGMENTATION_PROBAS, self.epochs)
