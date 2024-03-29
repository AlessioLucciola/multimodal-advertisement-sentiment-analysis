from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from shared.constants import FER_emotion_mapping

# FER Dataset
class FERDataset(Dataset):
    def __init__(self, 
                 data: pd.DataFrame, 
                 split: str, 
                 transform: any = None):
        self.data = data
        self.split = split
        self.transform = transform

        train_tfms, val_tfms = self.get_transformations()
        if self.transform is None:
            self.transform = train_tfms if split == 'train' else val_tfms
        self.tensor_transform = transforms.ToTensor()
        self.emotions = FER_emotion_mapping

        # Read the dataset
        if self.split == 'train':
            self.data = self.data.loc[self.data.Usage.isin(
                ['Training', 'PublicTest'])] # Train and val
            self.data.reset_index(drop=True, inplace=True)
            self.data = self.data.drop('Usage', axis=1)
        elif self.split == 'test':
            self.data = self.data.loc[self.data.Usage.isin(['PrivateTest'])] # Test
            self.data.reset_index(drop=True, inplace=True) 
            self.data = self.data.drop('Usage', axis=1)
        else:
            raise ValueError(
                "Invalid split type: must be either train or test")
        
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
        
    # Transformations
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
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = int(row['emotion'])
        img = np.copy(row[self.pix_cols].values.reshape(48, 48))
        img.setflags(write=True)

        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            img = self.tensor_transform(img)

        return img, img_id
    
