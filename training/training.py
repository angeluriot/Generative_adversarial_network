import os, pickle
from PIL import Image
import torch
from torch import nn

from training.settings import *
from training import data
from training.generator import *
from training.discriminator import *
from training import losses


class Trainer():

	def __init__(self, dataset: data.Dataset, generator: Generator, discriminator: Discriminator):

		self.dataset = dataset

		self.generator = generator
		self.discriminator = discriminator

		self.ma_generator = Generator().to(DEVICE)
		self.ma_generator.load_state_dict(self.generator.state_dict(), strict = True)

		gen_reg_ratio = PATH_LENGTH_INTERVAL / (PATH_LENGTH_INTERVAL + 1) if PATH_LENGTH else 1.0
		disc_reg_ratio = GRADIENT_PENALTY_INTERVAL / (GRADIENT_PENALTY_INTERVAL + 1) if GRADIENT_PENALTY else 1.0

		self.gen_optimizer = torch.optim.Adam(
			self.generator.parameters(),
			lr = LEARNING_RATE * gen_reg_ratio,
			betas = (BETA_1 ** gen_reg_ratio, BETA_2 ** gen_reg_ratio),
			eps = EPSILON
		)

		self.disc_optimizer = torch.optim.Adam(
			self.discriminator.parameters(),
			lr = LEARNING_RATE * disc_reg_ratio,
			betas = (BETA_1 ** disc_reg_ratio, BETA_2 ** disc_reg_ratio),
			eps = EPSILON
		)

		self.sample_z = self.ma_generator.gen_z(OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1])
		self.sample_noise = self.ma_generator.gen_noise(OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1])

		self.mean_path_length = torch.zeros((), device = DEVICE)
		self.step = 0


	# Save the models
	def save_models(self, path: str | list[str]) -> None:

		if type(path) == str:
			path = [path]

		for p in path:

			if not os.path.exists(p):
				os.makedirs(p)

			torch.save(self.generator.state_dict(), os.path.join(p, 'generator.pt'))
			torch.save(self.discriminator.state_dict(), os.path.join(p, 'discriminator.pt'))
			torch.save(self.ma_generator.state_dict(), os.path.join(p, 'ma_generator.pt'))
			torch.save(self.gen_optimizer.state_dict(), os.path.join(p, 'gen_optimizer.pt'))
			torch.save(self.disc_optimizer.state_dict(), os.path.join(p, 'disc_optimizer.pt'))
			pickle.dump(self.mean_path_length.item(), open(os.path.join(p, 'mean_path_length.pkl'), 'wb'))
			pickle.dump(self.step, open(os.path.join(p, 'step.pkl'), 'wb'))
			pickle.dump(self.sample_z, open(os.path.join(OUTPUT_DIR, 'sample_z.pkl'), 'wb'))
			pickle.dump(self.sample_noise, open(os.path.join(OUTPUT_DIR, 'sample_noise.pkl'), 'wb'))


	# Load the models
	def load_models(self, path: str) -> None:

		self.generator.load_state_dict(torch.load(os.path.join(path, 'generator.pt'), map_location = DEVICE))
		self.discriminator.load_state_dict(torch.load(os.path.join(path, 'discriminator.pt'), map_location = DEVICE))
		self.ma_generator.load_state_dict(torch.load(os.path.join(path, 'ma_generator.pt'), map_location = DEVICE))
		self.gen_optimizer.load_state_dict(torch.load(os.path.join(path, 'gen_optimizer.pt'), map_location = DEVICE))
		self.disc_optimizer.load_state_dict(torch.load(os.path.join(path, 'disc_optimizer.pt'), map_location = DEVICE))
		self.mean_path_length = torch.as_tensor(pickle.load(open(os.path.join(path, 'mean_path_length.pkl'), 'rb')), device = DEVICE)
		self.step = pickle.load(open(os.path.join(path, 'step.pkl'), 'rb')) + 1
		self.sample_z = pickle.load(open(os.path.join(OUTPUT_DIR, 'sample_z.pkl'), 'rb'))
		self.sample_noise = pickle.load(open(os.path.join(OUTPUT_DIR, 'sample_noise.pkl'), 'rb'))


	# Find previous session
	def find_previous_session(self) -> None:

		if os.path.exists(os.path.join(OUTPUT_DIR, 'last_model')):
			self.load_models(os.path.join(OUTPUT_DIR, 'last_model'))


	# Save samples
	def save_samples(self, path: str | list[str]) -> None:

		if type(path) == str:
			path = [path]

		images = self.ma_generator.z_to_images(self.sample_z, self.sample_noise)
		images = utils.create_grid(images, OUTPUT_SHAPE)
		images = Image.fromarray(images)

		for p in path:

			if not os.path.exists(os.path.dirname(p)):
				os.makedirs(os.path.dirname(p))

			images.save(p)


	# Moving average
	def moving_average(self) -> None:

		with torch.no_grad():

			for ma_p, p in zip(self.ma_generator.parameters(), self.generator.parameters()):
				ma_p.copy_(p.detach().lerp(ma_p, MA_BETA))

			for ma_b, b in zip(self.ma_generator.buffers(), self.generator.buffers()):
				ma_b.copy_(b.detach())

		self.ma_generator.eval()
		self.ma_generator.requires_grad_(False)


	# Print
	def print(self, gen_loss: float, path_length: float, disc_loss: float, grad_penalty: float) -> None:

		images_seen = self.step * BATCH_SIZE * ACCUMULATION_STEPS
		epochs = images_seen / self.dataset.size()

		print(f'Steps: {self.step:,} | Images: {images_seen:,} | Epochs: {epochs:.3f}   ||   ' + \
			f'Gen loss: {gen_loss:.3f} | PPL: {path_length / PATH_LENGTH_INTERVAL:.3f} (mean: {self.mean_path_length.item():.3f})   ||   ' + \
			f'Disc loss: {disc_loss:.4f} | Grad penalty: {grad_penalty / GRADIENT_PENALTY_INTERVAL:.4f}          ', end = '\r')


	# Clear gradients
	def clear_gradients(self) -> None:

		self.generator.zero_grad(set_to_none = True)
		self.discriminator.zero_grad(set_to_none = True)


	# Train the models
	def train(self) -> None:

		self.generator.train()
		self.discriminator.train()
		self.ma_generator.eval()

		self.generator.requires_grad_(False)
		self.discriminator.requires_grad_(False)
		self.ma_generator.requires_grad_(False)
		self.clear_gradients()

		print_path_length = 0.0
		print_grad_penalty = 0.0

		# Training loop
		while True:

			# Import data asynchronously
			all_real_images = [self.dataset.next().to(DEVICE, non_blocking = True) for _ in range(ACCUMULATION_STEPS)]

			# =============== TRAIN GENERATOR =============== #

			self.generator.requires_grad_(True)
			self.discriminator.requires_grad_(False)

			# ----- Main generator loss ----- #

			print_gen_loss = 0.0
			self.clear_gradients()

			# Accumulate gradients
			for _ in range(ACCUMULATION_STEPS):

				# Forward pass
				fake_images = self.generator(BATCH_SIZE)
				fake_scores = self.discriminator(fake_images)

				# Generator loss
				gen_loss = losses.gen_loss(fake_scores)
				print_gen_loss += gen_loss.item() / ACCUMULATION_STEPS

				# Backward pass
				gen_loss.backward()

			# Update weights
			self.generator.clean_nan()
			self.gen_optimizer.step()

			# ----- Path length regularization ----- #

			if PATH_LENGTH and self.step % PATH_LENGTH_INTERVAL == 0:

				print_path_length = 0.0
				self.clear_gradients()

				# Accumulate gradients
				for _ in range(ACCUMULATION_STEPS):

					# Forward pass
					path_batch_size = max(1, BATCH_SIZE // PATH_LENGTH_BATCH_SHRINK)
					fake_images, w = self.generator(path_batch_size, return_w = True)

					# Path length regularization
					self.generator.requires_grad_(False)
					path_loss, mean_path_length = losses.path_length(fake_images, w, self.mean_path_length)
					self.mean_path_length.copy_(mean_path_length.detach())
					print_path_length += path_loss.item() / ACCUMULATION_STEPS

					# Backward pass
					self.generator.requires_grad_(True)
					path_loss.backward()

				# Update weights
				self.generator.clean_nan()
				self.gen_optimizer.step()

			# =============== TRAIN DISCRIMINATOR =============== #

			self.generator.requires_grad_(False)
			self.discriminator.requires_grad_(True)

			# ----- Main discriminator loss ----- #

			print_disc_loss = 0.0
			self.clear_gradients()

			# Accumulate gradients
			for i in range(ACCUMULATION_STEPS):

				# Images
				fake_images = self.generator(BATCH_SIZE)
				real_images = all_real_images[i].detach()
				real_images.requires_grad = False

				# Forward pass
				fake_scores = self.discriminator(fake_images)
				real_scores = self.discriminator(real_images)

				# Discriminator loss
				disc_loss = losses.disc_loss(fake_scores, real_scores)
				print_disc_loss += disc_loss.item() / ACCUMULATION_STEPS

				# Backward pass
				disc_loss.backward()

			# Update weights
			self.discriminator.clean_nan()
			self.disc_optimizer.step()

			# ----- Gradient penalty ----- #

			if GRADIENT_PENALTY and self.step % GRADIENT_PENALTY_INTERVAL == 0:

				print_grad_penalty = 0.0
				self.clear_gradients()

				# Accumulate gradients
				for i in range(ACCUMULATION_STEPS):

					# Forward pass
					real_images = all_real_images[i].detach()
					real_images.requires_grad = True
					real_scores = self.discriminator(real_images)

					# Gradient penalty
					self.discriminator.requires_grad_(False)
					grad_penalty = losses.gradient_penalty(real_images, real_scores)
					print_grad_penalty += grad_penalty.item() / ACCUMULATION_STEPS

					# Backward pass
					self.discriminator.requires_grad_(True)
					grad_penalty.backward()

				# Update weights
				self.discriminator.clean_nan()
				self.disc_optimizer.step()

			# ======================================== #

			self.generator.requires_grad_(False)
			self.discriminator.requires_grad_(False)
			self.clear_gradients()

			# Moving average
			self.moving_average()

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

			# Print
			self.print(print_gen_loss, print_path_length, print_disc_loss, print_grad_penalty)

			# Update step
			self.step += 1
