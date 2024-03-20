import os

# Dataset configurations
DATASET_NAME = "fer2013"
DATA_DIR = "data/FER"
PATH_TO_SAVE_RESULTS = 'results'
METADATA_CSV = os.path.join(DATA_DIR, DATASET_NAME + ".csv")
SHUFFLE = True

# Train configurations
BATCH_SIZE = 64
N_EPOCHS = 200
LR = 1e-4
REG = 1e-3
MODEL_NAME = 'resnet18' # resnet18, resnet34, resnet50, resnet101, dense121, inception_v3, custom_cnn
SAVE_RESULTS = True
SAVE_MODELS = True

# Video configurations (FER2013 dataset)
NUM_CLASSES = 7
NUM_WORKERS = 1