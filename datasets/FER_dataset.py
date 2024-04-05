from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torchvision import transforms
from shared.constants import FER_emotion_mapping
from PIL import Image
import random
from collections import Counter

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class FERDataset(Dataset):
    def __init__(self, 
                 data: pd.DataFrame, 
                 is_train_dataset: bool = True,
                 apply_transformations: bool = True,
                 balance_dataset: bool = True,
                 augment_dataset: bool = True
                 ):
        self.data = data
        self.is_train_dataset = is_train_dataset
        self.apply_transformations = apply_transformations
        self.balance_dataset = balance_dataset
        self.augment_dataset = augment_dataset
        
        train_tfms, val_tfms = self.get_transformations()
        self.transformations = train_tfms if self.is_train_dataset else val_tfms            
        self.tensor_transform = transforms.ToTensor()
        self.emotions = FER_emotion_mapping
        # Reset index
        self.data.reset_index(drop=True, inplace=True)

        # Balance the dataset
        if self.balance_dataset and self.is_train_dataset: 
            data = self.apply_balance_dataset(data)
            data = data.drop(['balanced'], axis=1)

        # Augment the dataset
        if self.augment_dataset and self.is_train_dataset: 
            data = self.apply_data_augmentation(data, train_tfms)
            data = data.drop(['augmented'], axis=1)
            
        # Convert pixels to numpy array 
        pixels_values = [[int(i) for i in pix.split()]
                         for pix in self.data.pixels]   # For storing pixel values
        pixels_values = np.array(pixels_values)

        # Normalize pixel values to [0, 1]
        pixels_values = pixels_values/255.0
        self.data.drop(columns=['pixels'], axis=1, inplace=True)
        self.pix_cols = []  # For keeping track of column names  

        # Add pixel values to the dataframe as separate columns        
        for i in range(pixels_values.shape[1]):
            self.pix_cols.append(f'pixel_{i}') # Column name
            self.data[f'pixel_{i}'] = pixels_values[:, i] # Add pixel values to the dataframe

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
        print("--Data Balance-- Classes before balancing: ", data.emotion.value_counts().to_dict())
        emotion_counts = Counter(data.emotion)
        max_emotion, max_count = max(emotion_counts.items(), key=lambda x: x[1])
        print(f"--Data Balance-- The most common class is {max_emotion} with {max_count} images.")

        for emotion in data.emotion.unique():
            emotion_indices = data[data.emotion == emotion].index
            current_images = len(emotion_indices)

            if current_images < max_count:
                num_images_to_add = max_count - current_images
                print(f"--Data Balance (Oversampling)-- Adding {num_images_to_add} to {emotion} class..")
                aug_indices = random.choices(emotion_indices.tolist(), k=num_images_to_add)
                data = pd.concat([data, data.loc[aug_indices]])
                data.loc[aug_indices, "balanced"] = True
                emotion_indices = data[data["emotion"] == emotion].index
        data.fillna({"balanced": False}, inplace=True)
        print("--Data Balance-- Classes after balancing: ", data.emotion.value_counts().to_dict())

        return data
    
    def apply_data_augmentation(self, data, train_tfms):
        print("--Data Augmentation-- augment_data set to True. Training data will be augmented.")
        print("--Data Augmentation-- Classes before augmentation: ", data.emotion.value_counts().to_dict())

        # Apply transformations contained in train_tfms to the images
        data['augmented'] = False
        for idx, row in data.iterrows():
            img = np.copy(row[self.pix_cols].values.reshape(48, 48))
            img = Image.fromarray(img)
            img = train_tfms(img)
            img = np.array(img)
            data.loc[idx, self.pix_cols] = img.flatten()
            data.loc[idx, 'augmented'] = True

        print("--Data Augmentation-- Classes after augmentation: ", data.emotion.value_counts().to_dict())

        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = int(row['emotion'])
        img = np.copy(row[self.pix_cols].values.reshape(48, 48))
        img.setflags(write=True)

        # Apply transformations to the image if provided
        if self.transformations:
            img = Image.fromarray(img)
            img = self.transformations(img)
        else:
            img = self.tensor_transform(img)
            
        return img, img_id
    
