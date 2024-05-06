from torch.utils.data import Dataset
import torch
import pandas as pd


class DEAPDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ppg = torch.tensor(self.data["ppg"].iloc[index])
        valence = torch.tensor(self.data["valence"].iloc[index])

        return {"ppg": ppg,
                "valence": valence}
