from torch.utils.data import Dataset
import torch
import pandas as pd
from torch import nn
import torch.nn.functional as F


class GREXDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 is_train_dataset: bool = True):

        self.data = data
        self.is_train_dataset = is_train_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ppg = torch.tensor(self.data["ppg"].iloc[index])
        spatial_features = self.data["ppg_spatial_features"].iloc[index]
        valence = torch.tensor(self.data["val"].iloc[index])
        arousal = torch.tensor(self.data["aro"].iloc[index])
        # temporal_features = [torch.tensor(
        #     x) for x in self.data["ppg_temporal_features"].iloc[index]]

        # temporal_features = [F.pad(x, (0, 6 - len(x)))
        #                      for x in temporal_features]
        # temporal_features = nn.utils.rnn.pad_sequence(
        #     temporal_features, batch_first=True)
        temporal_features = torch.zeros((1, 1, 1))

        return {"ppg": ppg,
                "ppg_spatial_features": spatial_features,
                "ppg_temporal_features": temporal_features,
                "valence": valence,
                "arousal": arousal}
