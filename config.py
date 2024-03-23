import os

# Miscellanous configurations
RANDOM_SEED = 42
USE_DML = False
USE_MPS = False
USE_WANDB = True

# AUDIO
# Dataset configurations (RAVDESS dataset)
DATA_DIR = "data"
PATH_TO_SAVE_RESULTS = 'results'
AUDIO_FILES_DIR = os.path.join(DATA_DIR, "MERGED_AUDIO_FILES")
METADATA_CSV = os.path.join(DATA_DIR, "metadata.csv")
DF_SPLITTING = [0.80, 0.50] #[train/test splitting, test/val splitting]
LIMIT = None # Limit the number of samples in the dataset in percentage (0.5 means use only 50% of the dataset). Use "None" instead.

# Train configurations (RAVDESS dataset)
BATCH_SIZE = 32
N_EPOCHS = 500
LR = 1e-4
REG = 1e-3
SAVE_RESULTS = True
SAVE_MODELS = False
RESUME_TRAINING = False
PATH_MODEL_TO_RESUME = "path/to/model"
RESUME_EPOCH = 0

# Audio configurations (RAVDESS dataset)
AUDIO_SAMPLE_RATE = 48000
AUDIO_DURATION = 3
AUDIO_TRIM = 0.5
RAVDESS_NUM_CLASSES = 8

# ----------------------------

# VIDEO
# Dataset configurations (FER2013 dataset)
DATASET_NAME = "fer2013"
DATA_DIR = "data/VIDEO/FER"
PATH_TO_SAVE_RESULTS = 'results'
METADATA_CSV = os.path.join(DATA_DIR, DATASET_NAME + ".csv")
# DF_SPLITTING = [0.80, 0.50] #[train/test splitting, test/val splitting]
# LIMIT = None # Limit the number of samples in the dataset in percentage (0.5 means use only 50% of the dataset). Use "None" instead.
SHUFFLE = True

# Train configurations (FER2013 dataset)
BATCH_SIZE = 256
N_EPOCHS = 100
LR = 1e-5
REG = 1e-5
SAVE_RESULTS = True
SAVE_MODELS = False
MODEL_NAME = 'resnet18' # resnet18, resnet34, resnet50, resnet101, dense121, inception_v3, custom_cnn
# RESUME_TRAINING = False
# PATH_MODEL_TO_RESUME = "path/to/model"
# RESUME_EPOCH = 0

# Video configurations (FER2013 dataset)
NUM_CLASSES = 7
NUM_WORKERS = 1