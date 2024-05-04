from collections import Counter
from pathlib import Path
from torch.utils.data import Dataset
import random
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

pd.set_option('future.no_silent_downcasting', True) # Raise errors instead of warnings for pd.to_numeric

class ravdess_custom_dataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 files_dir: str,
                 is_train_dataset: bool = True,
                 preload_frames: bool = True,
                 balance_dataset: bool = True,
                 apply_transformations: bool = True,
                 normalize: bool = True,
                 ):
        self.data = data
        self.files_dir = Path(files_dir)
        self.is_train_dataset = is_train_dataset
        self.dataset_size = len(self.data)
        self.preload_frames_files = preload_frames
        self.apply_transformations = apply_transformations
        self.balance_dataset = balance_dataset
        self.normalize = normalize

        # Get transformations
        self.transformations = self.get_transformations()
        self.tensor_transform = transforms.ToTensor()

        # Balance the dataset
        if self.balance_dataset and self.is_train_dataset:
            self.data = self.apply_balance_dataset(self.data)

        # Preload frames files
        if self.preload_frames_files:
            self.frames = self.read_frames_files()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        frame_name = self.data.iloc[idx, 0]
        emotion = self.data.iloc[idx, 1]

        # Get frame from preloaded frames or load it from file
        if self.preload_frames_files:
            frame = self.frames[frame_name]
        else:
            frame = self.get_frame(frame_name)

        # Apply normalization
        if self.normalize:
            frame = self.normalize_frame(frame)

        # Apply transformations only to train balanced data (self.data.iloc[idx, -1]: balanced column)
        if self.apply_transformations and self.data.iloc[idx, -1]:
            frame = self.transformations(frame)
        else:
            frame = self.tensor_transform(frame)

        sample = {'frame': frame, 'emotion': emotion}

        return sample

    def get_frame(self, frame_name):
        frame_path = self.files_dir / frame_name
        with open(frame_path, 'rb') as f:
            frame = Image.open(f)
            frame = frame.convert('RGB') # Convert to RGB

        return frame
    
    def read_frames_files(self):
        print("--Data Preloading-- Preloading frames files.")
        frames = {}
        for frame_name in tqdm(self.data.iloc[:, 0]):
            frame = self.get_frame(frame_name)
            frames[frame_name] = frame

        return frames
    
    def get_transformations(self):
        train_tfms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor()
        ])
        val_tfms = transforms.Compose([
            transforms.ToTensor()
        ])
        
        return train_tfms if self.is_train_dataset else val_tfms
    
    def apply_balance_dataset(self, data):
        print("--Data Balance-- balance_data set to True. Training data will be balanced.")
        # Count images associated to each label
        labels_counts = Counter(data['emotion'])
        max_label, max_count = max(labels_counts.items(), key=lambda x: x[1])  # Majority class
        print(f"--Data Balance-- The most common class is {max_label} with {max_count} frames files.")
        
        # Balance the dataset by oversampling the minority classes
        for label in data['emotion'].unique():
            label_indices = data[data['emotion'] == label].index
            current_framess = len(label_indices)

            if current_framess < max_count:
                num_files_to_add = max_count - current_framess
                print(f"--Data Balance (Oversampling)-- Adding {num_files_to_add} to {label} class..")
                aug_indices = random.choices(label_indices.tolist(), k=num_files_to_add)
                self.data = pd.concat([data, data.loc[aug_indices]])
                # Apply data augmentation only to the augmented subset
                data.loc[aug_indices, "augmented"] = True
                label_indices = data[data["emotion"] == label].index
        data.fillna({"augmented": False}, inplace=True)

        return data
    
    def normalize_frame(self, frame):
        # Apply ImageNet normalization
        frame = self.tensor_transform(frame)
        frame = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(frame)
        frame = transforms.ToPILImage()(frame)

        return frame