import pickle
from utils.audio_utils import apply_AWGN, extract_waveform_from_audio_file, extract_zcr_features, extract_rmse_features, extract_mfcc_features, extract_features
from config import AUDIO_SAMPLE_RATE, AUDIO_OFFSET, AUDIO_DURATION, AUDIO_FILES_DIR, FRAME_LENGTH, HOP_LENGTH, USE_RAVDESS_ONLY, RAVDESS_FILES_DIR, NUM_MFCC
from collections import Counter
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

class RAVDESSCustomDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 files_dir: str,
                 transform: any = None,
                 is_train_dataset: bool = True,
                 balance_dataset: bool = True,
                 preload_audio_files: bool = True,
                 scale_audio_files: bool = True,
                 scaler: any = None
                 ):
        self.data = data
        self.files_dir = Path(files_dir)
        self.transform = transform
        self.is_train_dataset = is_train_dataset
        self.dataset_size = len(self.data)
        self.balance_dataset = balance_dataset
        self.preload_audio_files = preload_audio_files
        self.scale_audio_files = scale_audio_files
        self.scaler = scaler

        if self.balance_dataset and self.is_train_dataset:
            self.data = self.apply_balance_dataset(self.data)
        if self.is_train_dataset:
            print(f"--Dataset-- Training dataset size: {self.dataset_size}")

        if self.preload_audio_files:
            self.audio_files = self.read_audio_files()

        if self.scale_audio_files:
            if self.preload_audio_files:
                print("--Dataset-- Scaling audio files..")
                self.scale_data()
            else:
                print("--Dataset-- Preload audio files is set to False. Cannot scale audio files.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.preload_audio_files:
            audio_file = np.expand_dims(self.audio_files[self.data.iloc[idx, 0]], axis=0) # Add channel dimension to get a 4D tensor suitable for CNN
        else:
            waveform = self.get_waveform(os.path.join(RAVDESS_FILES_DIR if USE_RAVDESS_ONLY else AUDIO_FILES_DIR, self.data.iloc[idx, 0]))
            # Apply data augmentation if the audio file is marked as augmented
            if (self.balance_dataset and self.is_train_dataset):
                if (self.data.iloc[idx, 6] if USE_RAVDESS_ONLY else self.data.iloc[idx, 2]):
                    waveform = apply_AWGN(waveform)
            audio_file = np.expand_dims(self.get_audio_features(waveform), axis=0) # Add channel dimension to get a 4D tensor suitable for CNN
        emotion = self.data.iloc[idx, 1]

        sample = {'audio': audio_file, 'emotion': emotion}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_audio_features(self, audio):
        #return extract_features(waveform=audio, sample_rate=AUDIO_SAMPLE_RATE, n_mfcc=NUM_MFCC, n_fft=1024, win_length=512, n_mels=128, window='hamming', frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        return extract_mfcc_features(waveform=audio, sample_rate=AUDIO_SAMPLE_RATE, n_mfcc=NUM_MFCC, n_fft=1024, win_length=512, n_mels=128, window='hamming') 
    
    def get_waveform(self, audio):
        return extract_waveform_from_audio_file(audio, desired_length_seconds=AUDIO_DURATION, offset=AUDIO_OFFSET, desired_sample_rate=AUDIO_SAMPLE_RATE)
    
    def apply_balance_dataset(self, data):
        print("--Data Balance-- balance_data set to True. Training data will be balanced.")
        # Count images associated to each label
        labels_counts = Counter(self.data['emotion'])
        max_label, max_count = max(labels_counts.items(), key=lambda x: x[1])  # Majority class
        print(f"--Data Balance-- The most common class is {max_label} with {max_count} audio files.")
        
        # Balance the dataset by oversampling the minority classes
        for label in self.data['emotion'].unique():
            label_indices = self.data[self.data['emotion'] == label].index
            current_audios = len(label_indices)

            if current_audios < max_count:
                num_files_to_add = max_count - current_audios
                print(f"--Data Balance (Oversampling)-- Adding {num_files_to_add} to {label} class..")
                aug_indices = random.choices(label_indices.tolist(), k=num_files_to_add)
                self.metadata = pd.concat([self.data, self.data.loc[aug_indices]])
                # Apply data augmentation only to the augmented subset
                self.data.loc[aug_indices, "augmented"] = True
                label_indices = self.data[self.data["emotion"] == label].index
        self.data.fillna({"augmented": False}, inplace=True)

        return data
    
    def read_audio_files(self):
        waveform_dict = {}
        for i, audio_file in enumerate(tqdm(self.data.iloc[:, 0], desc="Loading audio files..", leave=False)):
            waveform = self.get_waveform(os.path.join(RAVDESS_FILES_DIR if USE_RAVDESS_ONLY else AUDIO_FILES_DIR, audio_file))
            if (self.balance_dataset and self.is_train_dataset):
                if (self.data.iloc[i, 6] if USE_RAVDESS_ONLY else self.data.iloc[i, 2]):
                    waveform = apply_AWGN(waveform)
            waveform = self.get_audio_features(waveform)
            waveform_dict[audio_file] = waveform
        return waveform_dict
    
    def scale_data(self):
        if self.is_train_dataset:
            scaler = StandardScaler()
            # Fit scaler on training data
            for waveform in self.audio_files.values():
                scaler.partial_fit(waveform.reshape(-1, waveform.shape[-1]))  # Reshape for scaler
            self.scaler = scaler
        # Scale each waveform in the dictionary
        scaled_audio_files = {}
        for filename, waveform in self.audio_files.items():
            scaled_waveform = self.scaler.transform(waveform.reshape(-1, waveform.shape[-1])).reshape(waveform.shape)
            scaled_audio_files[filename] = scaled_waveform

        self.audio_files = scaled_audio_files
            


