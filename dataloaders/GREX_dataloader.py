from torch.utils.data import DataLoader
from config import DATA_DIR
import os
import pickle
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.GREX_dataset import GREXDataset


class GREXDataLoader(DataLoader):
    def __init__(self, batch_size):
        self.batch_size = batch_size

        data_segments_path = os.path.join(
            DATA_DIR, "GREX", '3_Physio', 'Transformed')

        annotation_path = os.path.join(
            DATA_DIR, "GREX", '4_Annotation', 'Transformed')

        # NOTE: Important keys here are: "filt_PPG" and "raw_PPG". Sampling rate is 100.
        physio_trans_data_segments = pickle.load(
            open(os.path.join(data_segments_path, "physio_trans_data_segments.pickle"), "rb"))

        # NOTE: Important keys here are: 'ar_seg' and "vl_seg"
        annotations = pickle.load(
            open(os.path.join(annotation_path, "ann_trans_data_segments.pickle"), "rb"))

        self.ppg = torch.tensor(physio_trans_data_segments['filt_PPG'])
        self.valence = torch.tensor(annotations['vl_seg'])
        self.arousal = torch.tensor(annotations['ar_seg'])

        df = []
        for i in range(len(self.ppg)):
            df.append([self.ppg[i].tolist(), float(
                self.valence[i]), float(self.arousal[i])])

        self.data = pd.DataFrame(df)

        self.train_df, temp_df = train_test_split(self.data, test_size=0.2)
        self.val_df, self.test_df = train_test_split(temp_df, test_size=0.5)

    def get_train_dataloader(self):
        dataset = GREXDataset(self.train_df)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def get_val_dataloader(self):
        dataset = GREXDataset(self.val_df)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def get_test_dataloader(self):
        dataset = GREXDataset(self.test_df)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    # Test the dataloader
    dataloader = GREXDataLoader(batch_size=32)
    train_loader = dataloader.get_train_dataloader()
    val_loader = dataloader.get_val_dataloader()
    test_loader = dataloader.get_test_dataloader()

    for i, data in enumerate(train_loader):
        pgg, (val, ar) = data
        print(f"Batch {i}: {pgg.shape}, {val.shape}, {ar.shape}")
