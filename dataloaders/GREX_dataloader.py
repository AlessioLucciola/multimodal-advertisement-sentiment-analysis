from torch.utils.data import DataLoader
from config import DATA_DIR
import os
import pickle
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets.GREX_dataset import GREXDataset
from scipy.interpolate import CubicSpline


class GREXTransform:
    def __init__(self, df):
        self.df = df

    def jitter(self, data, sigma=0.05):
        noise = np.random.normal(loc=0., scale=sigma, size=data.shape)
        return data + noise

    def scaling(self, data, sigma=0.1):
        noise = np.random.normal(loc=1., scale=sigma, size=data.shape)
        return data * noise

    def magnitude_warping(self, data, sigma=0.2, knot=4):
        seq_len = data.shape[0]
        step = seq_len // knot
        # Get random curve
        control_points = np.concatenate((np.zeros(1), np.random.normal(
            loc=1.0, scale=sigma, size=(knot - 2)), np.zeros(1)))
        locs = np.arange(0, seq_len, step)

        # Apply cubic spline interpolation
        cs = CubicSpline(locs, control_points)
        return data * cs(np.arange(seq_len))

    def apply(self, item, p=0.5):
        transformations = [self.jitter, self.scaling, self.magnitude_warping]
        for transform in transformations:
            item = transform(item)
        return item

    def augment(self, n=5_000):
        augmented_data = []
        while len(augmented_data) < n:
            # Randomly select an item from the dataframe
            item = self.df.sample(1).iloc[0]
            ppg, val, aro = item["ppg"], item["val"], item["aro"]
            # Apply transformations and add to augmented_data
            new_item = {"ppg": self.apply(ppg), "val": val, "aro": aro}
            augmented_data.append(new_item)
        print(f"Augmented data: {len(augmented_data)}")
        self.df = pd.concat([self.df, pd.DataFrame(augmented_data)])
        return self.df


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

        self.ppg = (self.ppg - self.ppg.mean(dim=0, keepdim=True)) / \
            self.ppg.std(dim=0, keepdim=True)
        self.valence = torch.tensor(annotations['vl_seg']) - 1
        self.arousal = torch.tensor(annotations['ar_seg']) - 1

        print(
            f"valence counter: {np.unique(self.valence, return_counts=True)}")
        print(
            f"arousal counter: {np.unique(self.arousal, return_counts=True)}")

        df = []
        for i in range(len(self.ppg)):
            if (self.ppg[i] == 0.0).all():
                continue
            df.append({"ppg": self.ppg[i].numpy(), "val": float(
                self.valence[i]), "aro": float(self.arousal[i])})

        self.data = pd.DataFrame(df)

        self.train_df, temp_df = train_test_split(
            self.data, test_size=0.4, stratify=self.data[["val", "aro"]])
        self.val_df, self.test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df[["val", "aro"]])

        self.train_df = GREXTransform(self.train_df).augment()

        # self.train_df, temp_df = train_test_split(
        #     self.data, test_size=0.2)
        # self.val_df, self.test_df = train_test_split(
        #     temp_df, test_size=0.5)

        print(
            f"Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")

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
