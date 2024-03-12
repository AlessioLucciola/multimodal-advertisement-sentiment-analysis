from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd

# TO DO: It is necessary to handle the audio files. It must be transformed into a tensor with the audio wavelength.
class RAVDESSCustomDataset(Dataset):
    def __init__(self, csv_file, files_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.files_dir = Path(files_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_file = self.data.iloc[idx, 0] # Handle the audio file here
        emotion = self.data.iloc[idx, 1]
        emotion_intensity = self.data.iloc[idx, 2]
        statement = self.data.iloc[idx, 3]
        repetition = self.data.iloc[idx, 4]
        actor = self.data.iloc[idx, 5]

        sample = {'audio': audio_file, 'emotion': emotion, 'emotion_intensity': emotion_intensity,
                  'statement': statement, 'repetition': repetition, 'actor': actor}

        if self.transform:
            sample = self.transform(sample)

        return sample