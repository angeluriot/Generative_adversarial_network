import os, math
import torch


# ---------- Dataset ---------- #

DATA_DIR = 'D:/Datasets/Anime_faces' # Path to the dataset
IMAGE_SIZE = 256 # Width and height of the images
NB_CHANNELS = 3 # Number of channels in the images
FLIP_DATASET = True # Double the dataset by flipping the images
NB_WORKERS = 8 # Number of workers for the dataloader

# ---------- Outputs ---------- #

OUTPUT_DIR = './output' # Path to the output directory
OUTPUT_SHAPE = (8, 8) # Shape of the output image (columns, rows)
MODEL_SAVE_FREQUENCY = 1000 # Model save frequency (in steps)
SAMPLE_SAVE_FREQUENCY = 100 # Sample save frequency (in steps)

# ---------- Model ---------- #

LATENT_DIM = 512 # Dimension of the latent space
MAPPING_LAYERS = 8 # Number of layers in the mapping network

MIN_RESOLUTION = 4 # The smallest size of convolutional layers
GEN_MIN_FEATURES = 128 # The smallest number of features in the generator
GEN_MAX_FEATURES = 512 # The largest number of features in the generator
DIS_MIN_FEATURES = 128 # The smallest number of features in the discriminator
DIS_MAX_FEATURES = 512 # The largest number of features in the discriminator

KERNEL_SIZE = 3 # Size of the convolutional kernels
ALPHA = 0.2 # LeakyReLU slope
MINIBATCH_STD_GROUP_SIZE = 4 # Size of the groups for the minibatch standard deviation layer

# ---------- Training ---------- #

BATCH_SIZE = 8 # Batch size
ACCUMULATION_STEPS = 1 # Number of accumulation steps

LEARNING_RATE = 0.002 # Learning rate
MAPPING_LR_RATIO = 0.01 # Learning rate ratio of the mapping network
BETA_1 = 0.0 # Adam beta 1
BETA_2 = 0.99 # Adam beta 2
EPSILON = 10e-8 # Adam epsilon

STYLE_MIX_PROBA = 0.9 # Probability of mixing styles

PATH_LENGTH = True # Enable path length regularization
PATH_LENGTH_COEF = 2.0 # Path length regularization coefficient
PATH_LENGTH_DECAY = 0.01 # Path length regularization decay
PATH_LENGTH_BATCH_SHRINK = 2 # Batch size shrinking factor for path length regularization
PATH_LENGTH_INTERVAL = 4 # Interval of path length regularization

GRADIENT_PENALTY = True # Enable gradient penalty
GRADIENT_PENALTY_COEF = 10.0 # Gradient penalty coefficient
GRADIENT_PENALTY_INTERVAL = 16 # Interval of gradient penalty

MA_BETA = 0.9995 # Moving average generator beta

AUGMENTATION_PROBA = 0.0 # Probability of images modifications
PIXEL_AUGMENTATION = False # Pixel augmentation
GEOMETRIC_AUGMENTATION = False # Geometric augmentation

# ---------- Testing ---------- #

MEAN_W_SAMPLES = 10_000 # Number of samples to compute the mean W
MAPPING_BATCH_SIZE = 1000 # Batch size for the mapping network in testing
SYNTHESIS_BATCH_SIZE = BATCH_SIZE * 2 # Batch size for the synthesis network in testing

# ---------- Calculated ---------- #

GPU_ENABLED = torch.cuda.is_available()
DEVICE_NAME = 'cuda:0' if GPU_ENABLED else 'cpu'
DEVICE = torch.device(DEVICE_NAME)
SAMPLES_DIR = os.path.join(OUTPUT_DIR, 'images')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
MARGIN = IMAGE_SIZE // 8
NB_RESOLUTIONS = int(math.log(IMAGE_SIZE, 2)) - int(math.log(MIN_RESOLUTION, 2)) + 1
NB_W = 2 * NB_RESOLUTIONS
NB_NOISE = 2 * NB_RESOLUTIONS - 1
