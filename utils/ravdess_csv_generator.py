from config import RAVDESS_FILES, RAVDESS_CSV
import pandas as pd
import os

def create_ravdess_csv():
    file_names = []
    emotion = []
    emotion_intensity = []
    statement = []
    repetition = []
    actor = []
    for file in os.listdir(RAVDESS_FILES):
        file_names.append(file)
        file_info = file.split("-")
        emotion.append(int(file_info[2])-1)
        emotion_intensity.append(int(file_info[3])-1)
        statement.append(int(file_info[4])-1)
        repetition.append(int(file_info[5])-1)
        actor.append(int(file_info[6].split(".")[0])-1)

    data = {"file_name": file_names, "emotion": emotion, "emotion_intensity": emotion_intensity, "statement": statement, "repetition": repetition, "actor": actor}
    df = pd.DataFrame(data)
    df.to_csv(RAVDESS_CSV, index=False)

    return df

if __name__ == "__main__":
    create_ravdess_csv()