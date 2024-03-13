from pathlib import Path
from torch.utils.data import Dataset
from utils.audio_utils import extract_audio_features, extract_waveform_from_audio_file
from config import AUDIO_SAMPLE_RATE, AUDIO_TRIM, AUDIO_DURATION, RAVDESS_FILES
import pandas as pd
import numpy as np
import os

class RAVDESSCustomDataset(Dataset):
    def __init__(self, csv_file, files_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.files_dir = Path(files_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform = self.get_waveform(os.path.join(RAVDESS_FILES, self.data.iloc[idx, 0]))
        audio_file = np.expand_dims(self.get_audio_feature(waveform), axis=0) # Add channel dimension to get a 4D tensor suitable for CNN
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
    
    def get_audio_feature(self, audio):
        return extract_audio_features(audio, sample_rate=AUDIO_SAMPLE_RATE, n_mfcc=40, n_fft=1024, win_length=512, n_mels=128, window='hamming')
    
    def get_waveform(self, audio):
        return extract_waveform_from_audio_file(audio, desired_length_seconds=AUDIO_DURATION, trim_seconds=AUDIO_TRIM, desired_sample_rate=AUDIO_SAMPLE_RATE)