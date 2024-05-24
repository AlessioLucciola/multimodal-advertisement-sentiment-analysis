# Models configurations
AUDIO_MODEL_PATH = "AudioNetCT_2024-05-05_11-51-20"
VIDEO_MODEL_PATH ="VideoNet_vit-pretrained_2024-05-08_11-06-28_RAVDESS_WHITE_BACKGROUND_WITH_OVERLAP"
PPG_MODEL_PATH = "EmotionNet_2024-05-21_10-16-34_final"
AUDIO_MODEL_EPOCH = 155
VIDEO_MODEL_EPOCH = 100
PPG_MODEL_EPOCH = 565
AUDIO_IMPORTANCE = 0.6

# Audio/video configurations  
INPUT_DEVICE_INDEX = 1 # Alessio: 1 | Danilo: 1
OUTPUT_DEVICE_INDEX = 3 # Alessio: 3 | Danilo: 3
VIDEO_DEVICE_STREAM = '<video0>' # Alessio: '<video0>' | Danilo: '<video0>'

assert AUDIO_IMPORTANCE >= 0 and AUDIO_IMPORTANCE <= 1, "AUDIO_IMPORTANCE must be between 0 and 1"
