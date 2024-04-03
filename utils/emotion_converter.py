from enum import Enum


class Emotion(Enum):
    Angry = 0,
    Disgust = 1,
    Fear = 2,
    Happy = 3,
    Sad = 4,
    Surprise = 5,
    Neutral = 6,


def arousal_valence_to_emotion(arousal, valence):
    if arousal >= 0.5 and valence >= 0.5:
        return Emotion.Happy
    elif arousal >= 0.5 and valence < 0.5:
        return Emotion.Angry
    elif arousal < 0.5 and valence >= 0.5:
        return Emotion.Sad
    elif arousal < 0.5 and valence < 0.5:
        return Emotion.Neutral
    else:
        raise ValueError("Invalid arousal and valence values")
