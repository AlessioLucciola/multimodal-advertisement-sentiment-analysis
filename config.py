import os
# Common configurations
DATA_DIR = "data"
PATH_TO_SAVE_RESULTS = "results"
RANDOM_SEED = 42
# Limit the number of samples in the dataset in percentage (0.5 means use only 50% of the dataset). Use "None" instead.
LIMIT = None
# Balance the dataset if True, use the original dataset if False
BALANCE_DATASET = False

USE_DML = False
USE_MPS = True
USE_WANDB = False
SAVE_RESULTS = True
SAVE_MODELS = False
# EMOTION_NUM_CLASSES = 3  # Bad mood, neutral, good mood
EMOTION_NUM_CLASSES = 5  # Valence in [0,4]
RESUME_TRAINING = False
PATH_MODEL_TO_RESUME = "AudioNetCL_2024-03-28_17-13-18"
RESUME_EPOCH = 67
# ----------------------------

# Train configurations

LENGTH = 300
STEP = 250
WAVELET_STEP = 2
BATCH_SIZE = 128
N_EPOCHS = 1000
# LR = 0.00001
LR = 1e-3
REG = 0.1
DROPOUT_P = 0.5
#---
LSTM_ENC_HIDDEN = 8
LSTM_DEC_HIDDEN = 8
#---
# EmotionNet Transformer config
T_HEAD = 4
T_ENC_LAYERS = 6
T_DIM_FFW = 512
T_KERN = 4
T_STRIDE = 4
T_MAXPOOL = 0
MESSAGE = "Wavelet -> Slice + STFT"

# LOAD_DF = True
# SAVE_DF = False

LOAD_DF = False
SAVE_DF = False

# TODO: put this in the dataset configuration (GREX dataset)
AUGMENTATION_SIZE = 0
ADD_NOISE = False

# ----------------------------

# AUDIO
# Dataset configurations (RAVDESS dataset)
DATASET_NAME = "RAVDESS"  # Datasets: RAVDESS | ALL
DATASET_DIR = os.path.join(DATA_DIR, "AUDIO")
AUDIO_FILES_DIR = os.path.join(DATASET_DIR, "audio_merged_datasets_files")
RAVDESS_FILES_DIR = os.path.join(DATASET_DIR, "audio_ravdess_files")
METADATA_RAVDESS_CSV = os.path.join(DATASET_DIR, "audio_metadata_ravdess.csv")
METADATA_ALL_CSV = os.path.join(DATASET_DIR, "audio_metadata_all.csv")
DF_SPLITTING = [0.20, 0.50]  # [train/test splitting, test/val splitting]
USE_RAVDESS_ONLY = True  # Use only RAVDESS dataset if True, use all datasets if False
PRELOAD_AUDIO_FILES = True

# Audio configurations (RAVDESS dataset)
# AUDIO_SAMPLE_RATE = 48000
# AUDIO_DURATION = 3
# AUDIO_TRIM = 0.5
# RAVDESS_NUM_CLASSES = 8
# AUDIO_OFFSET = 0.5
# HOP_LENGTH = 512
# FRAME_LENGTH = 2048
# AUDIO_NUM_CLASSES = 8
# SCALE_AUDIO_FILES = True
# NUM_MFCC = 40
# LSTM_HIDDEN_SIZE = 128
# LSTM_NUM_LAYERS = 2

# # ----------------------------

# # VIDEO
# # Dataset configurations (FER dataset)
# DATASET_NAME = "fer2013"  # Datasets: fer2013 | ...
DATASET_DIR = os.path.join(DATA_DIR, "VIDEO/FER/")  # Dir: FER | ...
METADATA_CSV = os.path.join(DATASET_DIR, DATASET_NAME + ".csv")
VAL_SIZE = 0.2  # Validation size
SHUFFLE = True  # Shuffle the dataset
LIVE_TEST = False  # Test the model on live video if True, test on a video file if False
# Models: resnet18, resnet34, resnet50, resnet101, dense121, custom_cnn
MODEL_NAME = 'EmotionNet'

# Video configurations (FER dataset)
FER_NUM_CLASSES = 7  # Number of classes in the dataset (default: 7)
NUM_WORKERS = 1  # Number of workers for dataloader (default: 1)
