from config import DATA_DIR
import pandas as pd
import os

from shared.constants import CREMA_emotion_mapping, EMODB_emotion_mapping, EMOVO_emotion_mapping, RAVDESS_emotion_mapping, SAVEE_emotion_mapping, TESS_emotion_mapping, URDU_emotion_mapping, final_emotion_mapping

def create_datasets_csv():
    ravdess_df = create_ravdess_csv()
    crema_df = create_crema_csv()
    emodb_df = create_emodb_csv()
    emovo_df = create_emovo_csv()
    savee_df = create_savee_csv()
    tess_df = create_TESS_csv()
    urdu_df = create_URDU_csv()
    merged_df = pd.concat([ravdess_df, crema_df, emodb_df, emovo_df, savee_df, tess_df, urdu_df], ignore_index=True)
    merged_df = merged_df[~merged_df['emotion'].isin(['calm', 'boredom'])] # Remove calm and boredom emotions
    print(merged_df['emotion'].value_counts())
    merged_df['emotion'] = merged_df['emotion'].map(final_emotion_mapping)
    print(merged_df['emotion'].value_counts())
    merged_df.to_csv(os.path.join(DATA_DIR, "metadata.csv"), index=False)

def create_ravdess_csv():
    file_names = []
    emotions = []
    #emotion_intensity = []
    #statement = []
    #repetition = []
    #actor = []
    for file in os.listdir(os.path.join(DATA_DIR, "RAVDESS", "files")):
        file_names.append(file)
        file_info = file.split("-")
        emotions.append(RAVDESS_emotion_mapping[int(file_info[2])])
        #emotion_intensity.append(int(file_info[3])-1)
        #statement.append(int(file_info[4])-1)
        #repetition.append(int(file_info[5])-1)
        #actor.append(int(file_info[6].split(".")[0])-1)

    #data = {"file_name": file_names, "emotion": emotion, "emotion_intensity": emotion_intensity, "statement": statement, "repetition": repetition, "actor": actor}
    data = {"file_name": file_names, "emotion": emotions}
    df = pd.DataFrame(data)
    return df

def create_crema_csv():
    non_valid_files = ["1076_MTI_SAD_XX.wav", "1064_IEO_DIS_MD.wav"] # Incorrect files as pointed out by the dataset's authors
    file_names = []
    emotions = []
    for file in os.listdir(os.path.join(DATA_DIR, "CREMA", "AudioWAV")):
        if file not in non_valid_files:
            file_names.append(file)
            file_info = file.split("_")
            emotions.append(CREMA_emotion_mapping[file_info[2]])
    data = {"file_name": file_names, "emotion": emotions}
    df = pd.DataFrame(data)
    return df

def create_emodb_csv():
    file_names = []
    emotions = []
    for file in os.listdir(os.path.join(DATA_DIR, "EMO")):
        file_names.append(file)
        emotions.append(EMODB_emotion_mapping[file[5]])
    data = {"file_name": file_names, "emotion": emotions}
    df = pd.DataFrame(data)
    return df

def create_emovo_csv():
    file_names = []
    emotions = []
    for file in os.listdir(os.path.join(DATA_DIR, "EMOVO")):
        file_names.append(file)
        file_info = file.split("-")
        emotions.append(EMOVO_emotion_mapping[file_info[0]])
    data = {"file_name": file_names, "emotion": emotions}
    df = pd.DataFrame(data)
    return df

def create_savee_csv():
    file_names = []
    emotions = []
    for file in os.listdir(os.path.join(DATA_DIR, "SAVEE")):
        file_names.append(file)
        emotion = file[3]
        if (emotion == "s"):
            if (file[4] == "a"):
                emotion = "sa"
            else:
                emotion = "su"
        emotions.append(SAVEE_emotion_mapping[emotion])
    data = {"file_name": file_names, "emotion": emotions}
    df = pd.DataFrame(data)
    return df

def create_TESS_csv():
    file_names = []
    emotions = []
    for dir in os.listdir(os.path.join(DATA_DIR, "TESS")):
        if (os.path.isdir(os.path.join(DATA_DIR, "TESS", dir))):
            emotion = TESS_emotion_mapping[dir.split("_")[1]]
            for file in os.listdir(os.path.join(DATA_DIR, "TESS", dir)):
                file_names.append(file)
                emotions.append(emotion)
    data = {"file_name": file_names, "emotion": emotions}
    df = pd.DataFrame(data)
    return df

def create_URDU_csv():
    file_names = []
    emotions = []
    for dir in os.listdir(os.path.join(DATA_DIR, "URDU")):
        if (os.path.isdir(os.path.join(DATA_DIR, "URDU", dir))):
            emotion = URDU_emotion_mapping[dir]
            for file in os.listdir(os.path.join(DATA_DIR, "URDU", dir)):
                file_names.append(file)
                emotions.append(emotion)
    data = {"file_name": file_names, "emotion": emotions}
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    create_datasets_csv()

