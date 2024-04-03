from torch.utils.data import Dataset
import torch
import pandas as pd


class GREXDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 is_train_dataset: bool = True):

        self.data = data
        self.is_train_dataset = is_train_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ppg = torch.tensor(self.data.iloc[index, 0])

        valence = torch.tensor(self.data.iloc[index, 1])
        arousal = torch.tensor(self.data.iloc[index, 2])
        return (ppg, (valence, arousal))
