from torch.utils.data import Dataset
from utils.video_utils import get_transformations
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

        train_tfms, val_tfms = get_transformations()
        if self.transform is None:
            self.transform = train_tfms if split == 'train' else val_tfms
        self.tensor_transform = transforms.ToTensor()
        self.emotions = FER_emotion_mapping

        # read the dataset
        if self.split == 'train':
            self.data = self.data.loc[self.data.Usage.isin(
                ['Training', 'PublicTest'])]
            self.data.reset_index(drop=True, inplace=True)
            self.data = self.data.drop('Usage', axis=1)
        elif self.split == 'test':
            self.data = self.data.loc[self.data.Usage.isin(['PrivateTest'])]
            self.data.reset_index(drop=True, inplace=True)
            self.data = self.data.drop('Usage', axis=1)
        else:
            raise ValueError(
                "Invalid split type: must be either train or test")

#         pixels_values = []  # for storing pixel values
#         for pix in self.data.pixels:
#             values = [int(i) for i in pix.split()]
#             pixels_values.append(values)

#         pixels_values = np.array(pixels_values)
        
        
        pixels_values = [[int(i) for i in pix.split()]
                         for pix in self.data.pixels]   # for storing pixel values
        pixels_values = np.array(pixels_values)
        # rescaling pixel values
        pixels_values = pixels_values/255.0
        self.data.drop(columns=['pixels'], axis=1, inplace=True)
        self.pix_cols = []  # for keeping track of column names

        # add each pixel value as a column
        # # TODO: fix -> PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
        for i in range(pixels_values.shape[1]):
            self.pix_cols.append(f'pixel_{i}')
            
            self.data[f'pixel_{i}'] = pixels_values[:, i]

        self.df = self.data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = int(row['emotion'])
        img = np.copy(row[self.pix_cols].values.reshape(48, 48))
        img.setflags(write=True)

        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            img = self.tensor_transform(img)

        return img, img_id