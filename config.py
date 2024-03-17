import os

# Dataset configurations
DATA_DIR = "data"
PATH_TO_SAVE_RESULTS = 'results'
RAVDESS_DIR = os.path.join(DATA_DIR, "RAVDESS")
RAVDESS_CSV = os.path.join(RAVDESS_DIR, "ravdess.csv")
RAVDESS_FILES = os.path.join(RAVDESS_DIR, "files")
RAVDESS_DF_SPLITTING = [0.80, 0.50] #[train/test splitting, test/val splitting]

# Miscellanous configurations
RANDOM_SEED = 42
USE_DML = False
USE_MPS = False
USE_WANDB = False

# Train configurations
BATCH_SIZE = 32
N_EPOCHS = 500
LR = 1e-3
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