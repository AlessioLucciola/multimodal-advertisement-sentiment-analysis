from config import RAVDESS_VIDEO_FILES_DIR, RAVDESS_FRAMES_FILES_DIR, DATASET_DIR
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
    for file in os.listdir(RAVDESS_VIDEO_FILES_DIR):
        file_names.append(file)
        file_info = file.split("-")
        emotion.append(int(file_info[2])-1)
        emotion_intensity.append(int(file_info[3])-1)
        statement.append(int(file_info[4])-1)
        repetition.append(int(file_info[5])-1)
        actor.append(int(file_info[6].split(".")[0])-1)

    data = {"file_name": file_names, "emotion": emotion, "emotion_intensity": emotion_intensity, "statement": statement, "repetition": repetition, "actor": actor}
    df = pd.DataFrame(data)
    df.to_csv(DATASET_DIR + "/" + "ravdess_original" +".csv", index=False)

def create_ravdess_csv_from_frames(path):
    file_names = []
    emotion = []
    emotion_intensity = []
    statement = []
    repetition = []
    actor = []
    for file in os.listdir(os.path.join(RAVDESS_FRAMES_FILES_DIR, path)):
        file_names.append(file)
        file_info = file.split("_")[0].split("-")
        emotion.append(int(file_info[2])-1)
        emotion_intensity.append(int(file_info[3])-1)
        statement.append(int(file_info[4])-1)
        repetition.append(int(file_info[5])-1)
        actor.append(int(file_info[6])-1)

    data = {"file_name": file_names, "emotion": emotion, "emotion_intensity": emotion_intensity, "statement": statement, "repetition": repetition, "actor": actor}
    df = pd.DataFrame(data)
    df.to_csv(DATASET_DIR + "/" + "ravdess_frames" +".csv", index=False)

def create_ravdess_csv_from_frames_w_pixels(path):
    file_names = []
    emotion = []
    emotion_intensity = []
    statement = []
    repetition = []
    actor = []
    pixels_list = []
    for file in tqdm(os.listdir(os.path.join(RAVDESS_FRAMES_FILES_DIR, path))):
        file_names.append(file)
        file_info = file.split("_")[0].split("-")
        emotion.append(int(file_info[2])-1)
        emotion_intensity.append(int(file_info[3])-1)
        statement.append(int(file_info[4])-1)
        repetition.append(int(file_info[5])-1)
        actor.append(int(file_info[6])-1)
        # Generate pixels from image
        img = Image.open(os.path.join(RAVDESS_FRAMES_FILES_DIR, path, file))
        img = img.convert('L')
        img = img.resize((48, 48))
        img = np.array(img)
        img = img.flatten()
        img = ' '.join(str(p) for p in img)
        pixels_list.append(img)

    data = {"file_name": file_names, "emotion": emotion, "emotion_intensity": emotion_intensity, "statement": statement, "repetition": repetition, "actor": actor, "pixels": pixels_list}
    df = pd.DataFrame(data)
    df.to_csv(DATASET_DIR + "/" + frames_path +"_w_pixels.csv", index=False)

if __name__ == "__main__":
    frames_path = 'ravdess_frames_234x234' # ravdess_frames_48x48 | ravdess_frames_234x234
    # Comment / Uncomment the following lines to generate the desired CSV files
    create_ravdess_csv_from_video()
    create_ravdess_csv_from_frames(frames_path)
    create_ravdess_csv_from_frames_w_pixels(frames_path)
