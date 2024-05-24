from config import VIDEO_FILES_DIR, FRAMES_FILES_DIR, VIDEO_DATASET_DIR
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
import numpy as np

def create_ravdess_csv_from_video():
    file_names = []
    emotion = []
    emotion_intensity = []
    statement = []
    repetition = []
    actor = []
    for file in os.listdir(VIDEO_FILES_DIR):
        # For example: 01-01-03-01-01-02-05_72.png:
            # Modality  = 01 (01 = full-AV, 02 = video-only, 03 = audio-only)
            # Vocal channel = 01 (01 = speech, 02 = song)
            # Emotion = 03 (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
            # Emotion intensity = 01 (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
            # Statement = 01 (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
            # Repetition = 02 (01 = 1st repetition, 02 = 2nd repetition).
            # Actor = 05 (01 to 24. Odd numbered actors are male, even numbered actors are female).
        file_names.append(file)
        file_info = file.split("-")
        emotion.append(int(file_info[2])-1)
        emotion_intensity.append(int(file_info[3])-1)
        statement.append(int(file_info[4])-1)
        repetition.append(int(file_info[5])-1)
        actor.append(int(file_info[6].split(".")[0])-1)

    data = {"file_name": file_names, "emotion": emotion, "emotion_intensity": emotion_intensity, "statement": statement, "repetition": repetition, "actor": actor}
    df = pd.DataFrame(data)
    df.to_csv(VIDEO_DATASET_DIR + "/" + "RAVDESS_metadata_original" +".csv", index=False)

def create_ravdess_csv_from_frames(path):
    file_names = []
    emotion = []
    emotion_intensity = []
    statement = []
    repetition = []
    actor = []
    frame = []
    for file in os.listdir(FRAMES_FILES_DIR):
         # For example: 01-01-03-01-01-02-05_72.png:
            # Modality  = 01 (01 = full-AV, 02 = video-only, 03 = audio-only)
            # Vocal channel = 01 (01 = speech, 02 = song)
            # Emotion = 03 (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
            # Emotion intensity = 01 (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
            # Statement = 01 (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
            # Repetition = 02 (01 = 1st repetition, 02 = 2nd repetition).
            # Actor = 05 (01 to 24. Odd numbered actors are male, even numbered actors are female).
            # Frame = 72

        file_names.append(file)
        file_info = file.split("_")[0].split("-")
        emotion.append(int(file_info[2])-1)
        emotion_intensity.append(int(file_info[3])-1)
        statement.append(int(file_info[4])-1)
        repetition.append(int(file_info[5])-1)
        actor.append(int(file_info[6])-1)
        frame.append(int(file.split("_")[1].split(".")[0]))

    data = {"file_name": file_names, "emotion": emotion, "emotion_intensity": emotion_intensity, "statement": statement, "repetition": repetition, "actor": actor, "frame": frame}
    df = pd.DataFrame(data)
    df.to_csv(VIDEO_DATASET_DIR + "/" + path+".csv", index=False)

def create_ravdess_csv_from_frames_w_pixels(path):
    #  Retrieve size from path and convert into a tuple like (234, 234)
    img_size = tuple(map(int, path.split("_")[-1].split("x")))
    
    file_names = []
    emotion = []
    emotion_intensity = []
    statement = []
    repetition = []
    actor = []
    pixels_list = []
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
        file_names.append(file)
        file_info = file.split("_")[0].split("-")
        emotion.append(int(file_info[2])-1)
        emotion_intensity.append(int(file_info[3])-1)
        statement.append(int(file_info[4])-1)
        repetition.append(int(file_info[5])-1)
        actor.append(int(file_info[6])-1)
        # Generate pixels from image
        img = Image.open(os.path.join(FRAMES_FILES_DIR, path, file))
        img = img.convert('L')
        img = img.resize((img_size))
        img = np.array(img)
        img = img.flatten()
        img = ' '.join(str(p) for p in img)
        pixels_list.append(img)

    data = {"file_name": file_names, "emotion": emotion, "emotion_intensity": emotion_intensity, "statement": statement, "repetition": repetition, "actor": actor, "pixels": pixels_list}
    df = pd.DataFrame(data)
    df.to_csv(VIDEO_DATASET_DIR + "/" + path+".csv", index=False)

if __name__ == "__main__":
    frames_path = 'RAVDESS_metadata_frames' # RAVDESS_metadata_frames | RAVDESS_metadata_frames_w_pixels
    # Comment / Uncomment the following lines to generate the desired CSV files
    create_ravdess_csv_from_video()
    create_ravdess_csv_from_frames(frames_path)
    # create_ravdess_csv_from_frames_w_pixels(frames_path)
