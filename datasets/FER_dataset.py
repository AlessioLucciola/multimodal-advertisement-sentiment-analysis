from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torchvision import transforms
from shared.constants import FER_emotion_mapping
from PIL import Image
import random
from collections import Counter
from config import DATASET_DIR, DATASET_NAME

class FERDataset(Dataset):
    def __init__(self, 
                 data: pd.DataFrame, 
                 is_train_dataset: bool = True,
                 transform: any = None,
                 balance_dataset: bool = True,
                 augment_dataset: bool = True):
        self.data = data
        self.is_train_dataset = is_train_dataset
        self.transform = transform
        self.balance_dataset = balance_dataset
        self.augment_dataset = augment_dataset

        # Get transformations for training and validation
        train_tfms, val_tfms = self.get_transformations()
        if self.transform is None:
            self.transform = train_tfms if self.is_train_dataset else val_tfms
        
        self.tensor_transform = transforms.ToTensor()
        self.emotions = FER_emotion_mapping

        # Read the dataset
        if self.is_train_dataset:
            self.data = self.data.loc[self.data.Usage.isin(
                ['Training', 'PublicTest'])] # Train and val
            self.data.reset_index(drop=True, inplace=True)

            # Balance the dataset
            if self.balance_dataset: 
                self.data = self.apply_balance_dataset(self.data)
                self.data = self.data.drop(['balanced'], axis=1)

            # Augment the dataset
            if self.augment_dataset: 
                self.data = self.apply_data_augmentation(self.data, train_tfms)
                self.data = self.data.drop(['augmented'], axis=1)

            # Remove unnecessary columns
            self.data = self.data.drop(['Usage'], axis=1)
        else:
            self.data = self.data.loc[self.data.Usage.isin(['PrivateTest'])] # Test
            self.data.reset_index(drop=True, inplace=True) 
            self.data = self.data.drop('Usage', axis=1)
        
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
            # TODO: FIX THIS -> PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
            self.data[f'pixel_{i}'] = pixels_values[:, i] # Add pixel values to the dataframe
        
    def get_transformations(self):
        train_trans = [
            transforms.RandomCrop(48, padding=4, padding_mode='reflect'),
            transforms.RandomRotation(15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.01, 0.12),
                shear=(0.01, 0.03),
            ),
            transforms.RandomHorizontalFlip(),
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

        # Save augmented dataset to CSV
        data.to_csv(DATASET_DIR + DATASET_NAME + "_balanced.csv", index=False)

        return data
    
    def apply_data_augmentation(self, data, train_tfms):
        print("--Data Augmentation-- augment_data set to True. Training data will be augmented.")
        print("--Data Augmentation-- Classes before augmentation: ", data.emotion.value_counts().to_dict())

        # Apply transformations
        
        print("--Data Augmentation-- Classes after augmentation: ", data.emotion.value_counts().to_dict())

        # Save augmented dataset to CSV
        data.to_csv(DATASET_DIR + DATASET_NAME + "_augmented.csv", index=False)

        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = int(row['emotion'])
        img = np.copy(row[self.pix_cols].values.reshape(48, 48))
        img.setflags(write=True)

        # Apply transformations to the image if provided
        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            img = self.tensor_transform(img)

        return img, img_id
    
