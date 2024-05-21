# PPG constants
# CEAP_MEAN = -0.05
# CEAP_STD = 40.14

CEAP_MEAN = 0.546
CEAP_STD = 0.189

SCALED_DEAP_MEAN = 0.560787915300721
SCALED_DEAP_STD = 0.19942466308352924

# Video models
video_cnn_models_list = [
                'resnet18', 
                'resnet34', 
                'resnet50', 
                'resnet101', 
                'densenet121', 
                'custom-cnn',
                ]
                         
video_vit_models_list = [
                'vit-pretrained',
                ]

video_models_list = video_cnn_models_list + video_vit_models_list

# -----------------------------

# Mapping
RAVDESS_emotion_mapping = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprise"
}

RAVDESS_emotion_intensity_mapping = {
    0: "normal",
    1: "strong"
}

CREMA_emotion_mapping = {
    "NEU": "neutral",
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad",
    "DIS": "disgust",
    "FEA": "fearful"
}

EMODB_emotion_mapping = {
    "L": "boredom",
    "W": "angry",
    "F": "happy",
    "T": "sad",
    "E": "disgust",
    "A": "fearful",
    "N": "neutral"
}

EMOVO_emotion_mapping = {
    "neu": "neutral",
    "gio": "happy",
    "tri": "sad",
    "rab": "angry",
    "pau": "fearful",
    "dis": "disgust",
    "sor": "surprise"
}

SAVEE_emotion_mapping = {
    "n": "neutral",
    "h": "happy",
    "sa": "sad",
    "a": "angry",
    "f": "fearful",
    "d": "disgust",
    "su": "surprise"
}

TESS_emotion_mapping = {
    "neutral": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fearful",
    "disgust": "disgust",
    "surprise": "surprise"
}

URDU_emotion_mapping = {
    "Neutral": "neutral",
    "Happy": "happy",
    "Sad": "sad",
    "Angry": "angry",
}

final_emotion_mapping = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "disgust": 6,
    "surprise": 7
}

fer_emotion_mapping = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# -----------------------------

general_emotion_mapping = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprise"
}

mapping_to_positive_negative = {
    0: 0, # neutral -> neutral
    1: 0, # calm -> neutral
    2: 1, # happy -> positive
    3: 2, # sad -> negative
    4: 2, # angry -> negative
    5: 2, # fearful -> negative
    6: 2, # disgust -> negative
    7: 1 # surprise -> positive
}

merged_emotion_mapping = {
    0: "neutral",
    1: "positive",
    2: "negative"
}

ppg_emotion_mapping = {
    0: "negative",
    1: "neutral",
    2: "positive"
}
