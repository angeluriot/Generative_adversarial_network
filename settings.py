import os
import math

# Dataset
DATA_DIR = "<your dataset path>" # Path to the dataset
IMAGE_SIZE = 256 # Width and height of the images
NB_CHANNELS = 3 # Number of channels in the images
FLIP_DATASET = True # Double the dataset by flipping the images

# Outputs
OUTPUT_DIR = "./output" # Path to the output directory
OUTPUT_SHAPE = (7, 4) # Shape of the output image (columns, rows)
SAVE_FREQUENCY = 1000 # Save frequency (in steps)

# Model
LATENT_DIM = 512 # Dimension of the latent space
MAPPING_LAYERS = 8 # Number of layers in the mapping network
MIN_IMAGE_SIZE = 4 # The smallest size of convolutional layers
GEN_MIN_FILTERS = 64 # The smallest number of filters in the generator
GEN_MAX_FILTERS = 512 # The largest number of filters in the generator
DIS_MIN_FILTERS = 32 # The smallest number of filters in the discriminator
DIS_MAX_FILTERS = 512 # The largest number of filters in the discriminator
KERNEL_SIZE = 3 # Size of the convolutional kernels
ALPHA = 0.2 # LeakyReLU slope
GAIN = 1.2 # Equalized layers gain

# Training
BATCH_SIZE = 4 # Batch size
NB_EPOCHS = 10000 # Number of epochs
LEARNING_RATE = 0.002 # Learning rate
MAPPING_LR_RATIO = 0.01 # Learning rate ratio of the mapping network
BETA_1 = 0. # Adam beta 1
BETA_2 = 0.99 # Adam beta 2
EPSILON = 10e-8 # Adam epsilon
STYLE_MIX_PROBA = 0.9 # Probability of mixing styles
GRADIENT_PENALTY_COEF = 10. # Gradient penalty coefficient
GRADIENT_PENALTY_INTERVAL = 4 # Interval of gradient penalty
MA_HALF_LIFE = 10. # Moving average half life
AUGMENTATION_PROBA = 0.2 # Probability of images modifications
PIXEL_AUGMENTATION = True # Pixel augmentation
GEOMETRIC_AUGMENTATION = True # Geometric augmentation

# Calculated
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "images")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
MARGIN = IMAGE_SIZE // 8
MA_BETA = 0.5 ** (BATCH_SIZE / (MA_HALF_LIFE * 1000.)) if MA_HALF_LIFE > 0. else 0.
NB_BLOCKS = int(math.log(IMAGE_SIZE, 2)) - int(math.log(MIN_IMAGE_SIZE, 2)) + 1
