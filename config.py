import os

# Common configurations
DATA_DIR = "data"
PATH_TO_SAVE_RESULTS = "results"
RANDOM_SEED = 42

# Train / Validation / Test configurations
DF_SPLITTING = [0.20, 0.50] #[train/test splitting, test/val splitting]
LIMIT = None # Limit the number of samples in the dataset in percentage (0.5 means use only 50% of the dataset). Use "None" instead.
BALANCE_DATASET = True # Balance the dataset if True, use the original dataset if False

USE_DML = False
USE_MPS = False
USE_WANDB = False
SAVE_RESULTS = True
SAVE_MODELS = True

RESUME_TRAINING = False
PATH_MODEL_TO_RESUME = "VideoNet_vit-pretrained_2024-04-08_15-27-18"
RESUME_EPOCH = 66

# ----------------------------

# Train configurations
BATCH_SIZE = 256
N_EPOCHS = 50
LR = 1e-2
REG = 1e-2
DROPOUT_P = 0.2

# ----------------------------

# AUDIO
# Dataset configurations (RAVDESS dataset)
DATASET_NAME = "RAVDESS" # Datasets: RAVDESS | ALL
DATASET_DIR = os.path.join(DATA_DIR, "AUDIO")
AUDIO_FILES_DIR = os.path.join(DATASET_DIR, "audio_merged_datasets_files")
RAVDESS_FILES_DIR = os.path.join(DATASET_DIR, "audio_ravdess_files")
METADATA_RAVDESS_CSV = os.path.join(DATASET_DIR, "audio_metadata_ravdess.csv")
METADATA_ALL_CSV = os.path.join(DATASET_DIR, "audio_metadata_all.csv")
USE_RAVDESS_ONLY = True # Use only RAVDESS dataset if True, use all datasets if False
PRELOAD_AUDIO_FILES = True

# Audio configurations (RAVDESS dataset)
AUDIO_SAMPLE_RATE = 48000
AUDIO_DURATION = 3
AUDIO_TRIM = 0.5
RAVDESS_NUM_CLASSES = 8
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
VIDEO_DATASET_NAME = "ravdess_frames_48x48_w_pixels" # ravdess_original | ravdess_frames | ravdess_frames_48x48_w_pixels | ravdess_frames_234x234_w_pixels
DATASET_DIR = os.path.join(DATA_DIR, "VIDEO")   
RAVDESS_VIDEO_FILES_DIR = os.path.join(DATASET_DIR, "ravdess_video_files")
RAVDESS_FRAMES_FILES_DIR = os.path.join(DATASET_DIR, "ravdess_frames_files")
VIDEO_METADATA_CSV = os.path.join(DATASET_DIR, DATASET_NAME + ".csv") 
VIDEO_NUM_CLASSES = 8 # Number of classes in the dataset (default: 8)
NUM_WORKERS = 1 # Number of workers for dataloader (default: 1)

# Video configurations
MODEL_NAME = 'vit-pretrained' # Models: resnet18, resnet34, resnet50, resnet101, densenet121, custom-cnn, vit-pretrained
HIDDEN_SIZE = [512, 256, 128]  # Hidden layers configurations
IMG_SIZE = (48, 48) # (48, 48) | (224, 224)

# Train / Validation / Test configurations
APPLY_TRANSFORMATIONS = True # Apply transformations if True, use the original dataset if False
NORMALIZE = True # Normalize the images if True, use the original images if False
LIVE_TEST = False # Test the model on live video if True, test on a video file if False

# Fusion configurations
VIDEO_SAMPLE_RATE = 30
VIDEO_DURATION = 2
VIDEO_OVERLAPPING_SECONDS = 1.5

