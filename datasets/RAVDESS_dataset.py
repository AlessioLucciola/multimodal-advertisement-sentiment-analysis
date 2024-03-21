from pathlib import Path
from torch.utils.data import Dataset
from utils.audio_utils import extract_waveform_from_audio_file, extract_zcr_features, extract_rmse_features, extract_mfcc_features
from config import AUDIO_SAMPLE_RATE, AUDIO_OFFSET, AUDIO_DURATION, AUDIO_FILES_DIR, FRAME_LENGTH, HOP_LENGTH, USE_RAVDESS_ONLY, RAVDESS_FILES_DIR
import pandas as pd
import numpy as np
import os

class RAVDESSCustomDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 files_dir: str,
                 transform: any = None,
                 isTrainDataset: bool = True):
        self.data = data
        self.files_dir = Path(files_dir)
        self.transform = transform
        self.isTrain = isTrainDataset
        self.dataset_size = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform = self.get_waveform(os.path.join(RAVDESS_FILES_DIR if USE_RAVDESS_ONLY else AUDIO_FILES_DIR, self.data.iloc[idx, 0]))
        audio_file = np.expand_dims(self.get_audio_features(waveform), axis=0) # Add channel dimension to get a 4D tensor suitable for CNN
        emotion = self.data.iloc[idx, 1]

        sample = {'audio': audio_file, 'emotion': emotion}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_audio_features(self, audio):
        return extract_mfcc_features(waveform=audio, sample_rate=AUDIO_SAMPLE_RATE, n_mfcc=40, n_fft=1024, win_length=512, n_mels=128, window='hamming') 
    
    def get_waveform(self, audio):
        return extract_waveform_from_audio_file(audio, desired_length_seconds=AUDIO_DURATION, offset=AUDIO_OFFSET, desired_sample_rate=AUDIO_SAMPLE_RATE)