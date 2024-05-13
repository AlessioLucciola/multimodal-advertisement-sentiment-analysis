# Models configurations
AUDIO_MODEL_PATH = "AudioNetCT_2024-05-05_11-51-20"
VIDEO_MODEL_PATH = "VideoNet_vit-pretrained_2024-05-11_08-58-31_FER"
AUDIO_MODEL_EPOCH = 155
VIDEO_MODEL_EPOCH = 60
AUDIO_IMPORTANCE = 0.6

# Audio/video configurations  
INPUT_DEVICE_INDEX = 1 # Alessio: 1 | Danilo: 1
OUTPUT_DEVICE_INDEX = 3 # Alessio: 3 | Danilo: 3
VIDEO_DEVICE_STREAM = '<video0>' # Alessio: '<video0>' | Danilo: '<video0>'

assert AUDIO_IMPORTANCE >= 0 and AUDIO_IMPORTANCE <= 1, "AUDIO_IMPORTANCE must be between 0 and 1"