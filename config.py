import os

# Dataset configurations
DATA_DIR = "data"
RAVDESS_DIR = os.path.join(DATA_DIR, "RAVDESS")
RAVDESS_CSV = os.path.join(RAVDESS_DIR, "ravdess.csv")
RAVDESS_FILES = os.path.join(RAVDESS_DIR, "files")
RAVDESS_DF_SPLITTING = [0.8, 0.15] #[train/test splitting, test/val splitting]

# Train configurations
BATCH_SIZE = 32

# Audio configurations
AUDIO_SAMPLE_RATE = 48000
AUDIO_DURATION = 3
AUDIO_TRIM = 0.5