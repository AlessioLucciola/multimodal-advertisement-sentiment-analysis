import os

# Miscellanous configurations
RANDOM_SEED = 42
USE_DML = False
USE_MPS = False
USE_WANDB = False

# AUDIO
# Dataset configurations (RAVDESS dataset)
DATA_DIR = "data"
PATH_TO_SAVE_RESULTS = "results"
AUDIO_FILES_DIR = os.path.join(DATA_DIR, "audio_merged_datasets_files")
RAVDESS_FILES_DIR = os.path.join(DATA_DIR, "audio_ravdess_files")
METADATA_RAVDESS_CSV = os.path.join(DATA_DIR, "audio_metadata_ravdess.csv")
METADATA_ALL_CSV = os.path.join(DATA_DIR, "audio_metadata_all.csv")
DF_SPLITTING = [0.20, 0.50] #[train/test splitting, test/val splitting]
LIMIT = None # Limit the number of samples in the dataset in percentage (0.5 means use only 50% of the dataset). Use "None" instead.
USE_RAVDESS_ONLY = True # Use only RAVDESS dataset if True, use all datasets if False
BALANCE_DATASET = True # Balance the dataset if True, use the original dataset if False
PRELOAD_AUDIO_FILES = True

# Train configurations (RAVDESS dataset)
BATCH_SIZE = 32
N_EPOCHS = 500
LR = 1e-3
REG = 1e-3
DROPOUT_P = 0.2
SAVE_RESULTS = True
SAVE_MODELS = True
RESUME_TRAINING = True
PATH_MODEL_TO_RESUME = "AudioNetCL_2024-03-28_17-13-18"
RESUME_EPOCH = 67

# Audio configurations (RAVDESS dataset)
AUDIO_SAMPLE_RATE = 48000
AUDIO_DURATION = 3
AUDIO_TRIM = 0.5
RAVDESS_NUM_CLASSES = 8
AUDIO_OFFSET = 0.5
HOP_LENGTH = 512
FRAME_LENGTH = 2048
NUM_CLASSES = 8
SCALE_AUDIO_FILES = True
NUM_MFCC = 40
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2

# ----------------------------

# VIDEO
# Dataset configurations (FER2013 dataset)
DATASET_NAME = "fer2013" # Datasets: fer2013
DATA_DIR = "data/VIDEO/FER" # Directory: data/VIDEO/FER
PATH_TO_SAVE_RESULTS = 'results'
METADATA_CSV = os.path.join(DATA_DIR, DATASET_NAME + ".csv")
VAL_SIZE = 0.2 # Validation size
LIMIT = None # Limit the number of samples in the dataset in percentage (0.5 means use only 50% of the dataset). Use "None" instead.
SHUFFLE = True # Shuffle the dataset
BALANCE_DATASET = False # Balance the dataset if True, use the original dataset if False
LIVE_TEST = True # Test the model on live video if True, test on a video file if False

# Train configurations (FER2013 dataset)
BATCH_SIZE = 512 # Default: 64
N_EPOCHS = 100
LR = 1e-5
REG = 1e-5
DROPOUT_P = 0.2
SAVE_RESULTS = True
SAVE_MODELS = True
MODEL_NAME = 'resnet50' # Models: resnet18, resnet34, resnet50, resnet101, dense121, inception_v3, custom_cnn
RESUME_TRAINING = False
PATH_MODEL_TO_RESUME = "path/to/model"
RESUME_EPOCH = 0

# Video configurations (FER2013 dataset)
NUM_CLASSES = 7 # Number of classes in the dataset (default: 7)
NUM_WORKERS = 1 # Number of workers for dataloader (default: 1)
