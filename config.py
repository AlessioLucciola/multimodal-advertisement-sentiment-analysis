from typing import List
import os

# General configurations
DATA_DIR = "data"
PATH_TO_SAVE_RESULTS = "results"
RANDOM_SEED = 42
USE_DML = False
USE_MPS = False
USE_WANDB = False
SAVE_RESULTS = True
SAVE_MODELS = True

# Dataset configurations
DATASET_NAME: str = "RAVDESS" # Datasets: RAVDESS | ALL
DF_SPLITTING: List = [0.20, 0.50] #[train/test splitting, test/val splitting]
LIMIT: int | str = None # Limit the number of samples in the dataset in percentage (0.5 means use only 50% of the dataset). Use "None" instead.
BALANCE_DATASET: bool = True # Balance the dataset if True, use the original dataset if False
USE_POSITIVE_NEGATIVE_LABELS: bool = True
NUM_CLASSES: int = 3 if USE_POSITIVE_NEGATIVE_LABELS else 8 # Number of classes in the dataset (default: 8)


# Test configurations
PATH_MODEL_TO_TEST = "AudioNetCL_2024-04-17_10-28-10"
TEST_EPOCH = 30 # Number of epoch to test or "best" to test the best model

# Resume training configurations
RESUME_TRAINING = False
PATH_MODEL_TO_RESUME = "VideoNet_vit-pretrained_2024-04-08_15-27-18"
RESUME_EPOCH = 66

# Train configurations
BATCH_SIZE = 32 # Max (for ViT): 512 | Max (for CNN): 32
N_EPOCHS = 30
LR = 1e-3
REG = 1e-3
DROPOUT_P = 0.2

# ----------------------------

# AUDIO
# Dataset configurations (RAVDESS dataset)
AUDIO_DATASET_DIR = os.path.join(DATA_DIR, "AUDIO")
AUDIO_FILES_DIR = os.path.join(AUDIO_DATASET_DIR, "audio_merged_datasets_files")
AUDIO_RAVDESS_FILES_DIR = os.path.join(AUDIO_DATASET_DIR, "audio_ravdess_files")
AUDIO_METADATA_RAVDESS_CSV = os.path.join(AUDIO_DATASET_DIR, "audio_metadata_ravdess.csv")
AUDIO_METADATA_ALL_CSV = os.path.join(AUDIO_DATASET_DIR, "audio_metadata_all.csv")
USE_RAVDESS_ONLY = True # Use only RAVDESS dataset if True, use all datasets if False
PRELOAD_AUDIO_FILES = True

# Audio configurations (RAVDESS dataset)
AUDIO_SAMPLE_RATE = 48000
AUDIO_DURATION = 3
AUDIO_TRIM = 0.5
AUDIO_OFFSET = 0.5
HOP_LENGTH = 512
FRAME_LENGTH = 2048
AUDIO_NUM_CLASSES = 8
SCALE_AUDIO_FILES = True
NUM_MFCC = 40
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2

# ----------------------------

# VIDEO 
# Dataset configurations
VIDEO_DATASET_NAME = "ravdess" 
VIDEO_DATASET_DIR = os.path.join(DATA_DIR, "VIDEO")   
VIDEO_FILES_DIR = os.path.join(VIDEO_DATASET_DIR, "ravdess_video_files")
FRAMES_FILES_DIR = os.path.join(VIDEO_DATASET_DIR, "ravdess_frames_files")
VIDEO_METADATA_CSV = os.path.join(VIDEO_DATASET_DIR, VIDEO_DATASET_NAME + "_original.csv") 
VIDEO_METADATA_FRAMES_CSV = os.path.join(VIDEO_DATASET_DIR, VIDEO_DATASET_NAME + "_frames.csv")

# Models configurations
MODEL_NAME = 'resnet34' # Models: resnet18, resnet34, resnet50, resnet101, densenet121, custom-cnn, vit-pretrained
HIDDEN_SIZE = [512, 256, 128]  # Hidden layers configurations
IMG_SIZE = (224, 224)
NUM_WORKERS = os.cpu_count() # Number of workers for dataloader, set to 0 if you want to run the code in a single process

# Train / Validation / Test configurations
OVERLAP_SUBJETS_FRAMES = False # Overlap the frames of the subjects between train, validation and test if True, False otherwise
USE_DF_SPLITTING = False # If OVERLAP_SUBJETS_FRAMES = False, split the dataset using the DF_SPLITTING configuration if True, n-1 subjects for training, 1 for validation and 1 for testing if False
PRELOAD_FRAMES = True # Preload frames if True, load frames on the fly if False
APPLY_TRANSFORMATIONS = True # Apply transformations if True, use the original dataset if False
NORMALIZE = True # Normalize the images if True, use the original images if False
USE_VIDEO = False # Use test video or live video if True, use test dataset if False
LIVE_VIDEO = True # If USE_VIDEO = True, use live video if True, use offline video test file if False
VIDEO_OFFLINE_FILE = os.path.join(VIDEO_DATASET_DIR, "test_video.mp4") # Offline video file

# Fusion configurations
VIDEO_DURATION = 2
