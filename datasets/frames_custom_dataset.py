from collections import Counter
from pathlib import Path
from torch.utils.data import Dataset
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import transforms


class frames_custom_dataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 files_dir: str,
                 is_train_dataset: bool = True,
                 preload_frames: bool = True,
                 apply_transformations: bool = True,
                 balance_dataset: bool = True,
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
        train_tfms, val_tfms = self.get_transformations()
        self.transformations = train_tfms if self.is_train_dataset else val_tfms
        self.tensor_transform = transforms.ToTensor()

        # Balance the dataset
        if self.balance_dataset and self.is_train_dataset:
            self.data = self.apply_balance_dataset(self.data)
        if self.is_train_dataset:
            print(f"--Dataset-- Training dataset size: {self.dataset_size}")

        # Preload frames
        if self.preload_frames_files:
            self.frames = self.read_frames()

        # Normalize frames
        if self.normalize:
            print("--Data Normalization-- normalize set to True. Applying normalization to the images.")
            self.data['frame'] = self.data['frame'].apply(lambda x: x / 255.0)
        else:
            self.data['frame'] = self.data['frame'].apply(lambda x: np.array(x, dtype=np.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.preload_frames_files:
            frame = np.expand_dims(self.frames[self.data.iloc[idx, 0]], axis=0) # Add channel dimension to get a 4D tensor suitable for CNN
        else:
            frame = np.load(self.files_dir / self.data.iloc[idx, 0])
        emotion = self.data.iloc[idx, 1]

        if self.apply_transformations:
            frame = self.transformations(frame)
        else:
            frame = self.tensor_transform(frame)

        return frame, emotion

    def get_transformations(self):
        if self.apply_transformations:
            print("--Data Transformations-- apply_transformations set to True. Applying transformations to the images.")
            train_trans = [
                transforms.RandomCrop(48, padding=4, padding_mode='reflect'), # Random crop
                transforms.RandomRotation(15), # Random rotation
                transforms.RandomAffine( # Random affine transformation
                    degrees=0,
                    translate=(0.01, 0.12),
                    shear=(0.01, 0.03),
                ),
                transforms.RandomHorizontalFlip(), # Random horizontal flip
                transforms.ToTensor(),
            ]
        else:
            train_trans = [
                transforms.ToTensor(),
            ]

        val_trans = [
            transforms.ToTensor(),
        ]

        train_transforms = transforms.Compose(train_trans)
        valid_transforms = transforms.Compose(val_trans)

        return train_transforms, valid_transforms
    
    def apply_balance_dataset(self, data):
        print("--Data Balance-- balance_data set to True. Training data will be balanced.")
        # Count images associated to each label
        labels_counts = Counter(self.data['emotion'])
        max_label, max_count = max(labels_counts.items(), key=lambda x: x[1])  # Majority class
        print(f"--Data Balance-- The most common class is {max_label} with {max_count} frames.")
        
        # Balance the dataset by oversampling the minority classes
        for label in self.data['emotion'].unique():
            label_indices = self.data[self.data['emotion'] == label].index
            current_frames = len(label_indices)

            if current_frames < max_count:
                num_files_to_add = max_count - current_frames
                print(f"--Data Balance (Oversampling)-- Adding {num_files_to_add} to {label} class..")
                aug_indices = random.choices(label_indices.tolist(), k=num_files_to_add)
                self.metadata = pd.concat([self.data, self.data.loc[aug_indices]])
                # Apply data augmentation only to the augmented subset
                self.data.loc[aug_indices, "augmented"] = True
                label_indices = self.data[self.data["emotion"] == label].index
        self.data.fillna({"augmented": False}, inplace=True)

        return data
    
    def read_frames(self):
        frames = {}
        for idx in tqdm(range(self.dataset_size)):
            frame = np.load(self.files_dir / self.data.iloc[idx, 0])
            frames[self.data.iloc[idx, 0]] = frame
        return frames