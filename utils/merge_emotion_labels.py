import os
import pandas as pd
from PIL import Image
import numpy as np
from config import FRAMES_FILES_DIR, VIDEO_DATASET_DIR
from tqdm import tqdm
from utils.ravdess_csv_generator import create_ravdess_csv_from_frames
import shutil


def merge_emotion_labels():
    file_names = []
    new_file_names = []
    emotion = []
    new_emotion = []
    emotion_intensity = []
    statement = []
    repetition = []
    actor = []
    frame = []
    for file in tqdm(os.listdir(FRAMES_FILES_DIR)):
        # For example: 01-01-03-01-01-02-05_72.png:
            # Modality  = 01 (01 = full-AV, 02 = video-only, 03 = audio-only)
            # Vocal channel = 01 (01 = speech, 02 = song)
            # Emotion = 03 (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
            # Emotion intensity = 01 (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
            # Statement = 01 (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
            # Repetition = 02 (01 = 1st repetition, 02 = 2nd repetition).
            # Actor = 05 (01 to 24. Odd numbered actors are male, even numbered actors are female).
            # Frame = 72
        # Become 01-01-02-01-01-02-05_72.png, where the emotion is remapped to neutral (0), positive (1) and negative (2)
        # 0: neutral, calm
        # 1: happy, surprise
        # 2: angry, fearful, disgust, sad

        # Retrieve the emotion label from the file name
        file_names.append(file)
        file_info = file.split("_")[0].split("-")
        emotion.append(int(file_info[2])-1)
        emotion_intensity.append(int(file_info[3])-1)
        statement.append(int(file_info[4])-1)
        repetition.append(int(file_info[5])-1)
        actor.append(int(file_info[6])-1)
        frame.append(int(file.split("_")[1].split(".")[0]))
        
        # Remap "Emotion" value using the merged_emotion_mapping
        if emotion[-1] in [0, 1]:
            new_emotion.append(0)
        elif emotion[-1] in [2, 7]:
            new_emotion.append(1)
        else:
            new_emotion.append(2)   

        # Create a new file name for the remapped frames
        new_file_name = f"{file_info[0]}-{file_info[1]}-0{new_emotion[-1]+1}-{file_info[3]}-{file_info[4]}-{file_info[5]}-{file_info[6]}_{frame[-1]}.png"
        new_file_names.append(new_file_name)

    # Create a new CSV file with the remapped emotions
    data = {"file_name": file_names, "emotion": new_emotion, "emotion_intensity": emotion_intensity, "statement": statement, "repetition": repetition, "actor": actor, "frame": frame}
    df = pd.DataFrame(data)
    df.to_csv(VIDEO_DATASET_DIR + "/" + "_metadata_frames_remapped.csv", index=False)

if __name__ == "__main__":
    # Comment / Uncomment the following lines to generate the desired CSV files
    merge_emotion_labels()