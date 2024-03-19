import os

# Dataset configurations
DATA_DIR = "data"
PATH_TO_SAVE_RESULTS = 'results'
AUDIO_FILES_DIR = os.path.join(DATA_DIR, "MERGED_AUDIO_FILES")
METADATA_CSV = os.path.join(DATA_DIR, "metadata.csv")
DF_SPLITTING = [0.80, 0.50] #[train/test splitting, test/val splitting]
LIMIT = None # Limit the number of samples in the dataset in percentage (0.5 means use only 50% of the dataset). Use "None" instead.

# Miscellanous configurations
RANDOM_SEED = 42
USE_DML = True
USE_MPS = False
USE_WANDB = False

# Train configurations
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