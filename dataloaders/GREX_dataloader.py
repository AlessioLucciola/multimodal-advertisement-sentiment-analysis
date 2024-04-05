import json
from torch.utils.data import DataLoader
from config import AUGMENTATION_SIZE, BALANCE_DATASET, DATA_DIR, MODEL_NAME, RANDOM_SEED
import os
import pickle
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets.GREX_dataset import GREXDataset
from scipy.interpolate import CubicSpline
from utils.ppg_utils import extract_ppg_features_from_df


class GREXTransform:
    def __init__(self, df):
        self.df = df
        self.transformations = set()

    def jitter(self, data):
        sigma_min, sigma_max = 0.01, 0.2
        sigma = np.random.uniform(sigma_min, sigma_max)
        noise = np.random.normal(loc=0., scale=sigma, size=data.shape)
        return data + noise

    def scaling(self, data):
        sigma_min, sigma_max = 0.01, 0.3
        sigma = np.random.uniform(sigma_min, sigma_max)
        noise = np.random.normal(loc=1., scale=sigma, size=data.shape)
        return data * noise

    def magnitude_warping(self, data):
        sigma_min, sigma_max = 0.1, 0.3
        knot_min, knot_max = 4, 6
        sigma = np.random.uniform(sigma_min, sigma_max)
        knot = np.random.randint(knot_min, knot_max)
        seq_len = data.shape[0]
        step = seq_len // knot
        # Get random curve
        control_points = np.concatenate((np.zeros(1), np.random.normal(
            loc=1.0, scale=sigma, size=(knot - 2)), np.zeros(1)))
        locs = np.arange(0, seq_len, step)

        # Apply cubic spline interpolation
        cs = CubicSpline(locs, control_points)
        return data * cs(np.arange(seq_len))

    def time_shifting(self, data):
        shift = np.random.randint(low=-1000, high=1000)
        return np.roll(data, shift)

    # def window_slicing(self, data):
    #     start = np.random.randint(low=0, high=data.shape[0]//2)
    #     end = start + \
    #         np.random.randint(low=data.shape[0]//2, high=data.shape[0])
    #     sliced_data = data[start:end]

    #     # Calculate the number of zeros to add
    #     pad_length = 2000 - len(sliced_data)

    #     # Pad the sliced data with zeros at the end
    #     padded_data = np.pad(sliced_data, (0, pad_length))

    #     return padded_data

    # def flipping(self, data):
    #     return -data

    def apply(self, item, p=0.5):
        self.transformations = {self.jitter,
                                self.scaling, self.magnitude_warping}

        # subset = [item for item in self.transformations if np.random.rand() < p]
        for transform in self.transformations:
            item = transform(item)
        return item

    def augment(self, n=5_000):
        augmented_data = []
        print(f"Original data: {len(augmented_data)}")
        while len(augmented_data) < n:
            # Randomly select an item from the dataframe
            item = self.df.sample(1).iloc[0]
            ppg, val, aro = item["ppg"], item["val"], item["aro"]
            # Apply transformations and add to augmented_data
            new_item = {"ppg": self.apply(ppg), "val": val, "aro": aro}
            augmented_data.append(new_item)
        print(f"Augmented data: {len(augmented_data)}")
        df = pd.concat([self.df, pd.DataFrame(augmented_data)])
        return df

    def balance(self):
        for class_name in ["val", "aro"]:
            class_counts = self.df[class_name].value_counts()
            max_count = class_counts.max()
            while not all(class_counts == max_count):
                for cls in class_counts.index:
                    cls_count = class_counts[cls]
                    if cls_count < max_count:
                        cls_df = self.df[self.df[class_name] == cls]
                        n_samples = max_count - cls_count
                        samples = cls_df.sample(n_samples, replace=True)
                        # Apply transformations to the new samples
                        samples["ppg"] = samples["ppg"].apply(self.apply)
                        self.df = pd.concat([self.df, samples])
                        class_counts = self.df[class_name].value_counts()
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

        # physio_trans_data_session = pickle.load(
        #     open(os.path.join(data_segments_path, "physio_trans_data_session.pickle"), "rb"))

        # with open('session.json', 'w') as f:
        #     json.dump(physio_trans_data_session, f, default=str)
        # with open('segments.json', 'w') as f:
        #     json.dump(physio_trans_data_segments, f, default=str)

        # session_ppg = physio_trans_data_session['filt_PPG']
        # segments_ppg = physio_trans_data_segments['filt_PPG']
        # print(f"Session PPG: {len(session_ppg)}")
        # print(f"Segment PPG: {len(segments_ppg)}")
        # for session in session_ppg:
        #     print(f"Session length: {len(session)}")
        # for segment in segments_ppg:
        #     print(f"Segment length: {len(segment)}")
        # raise ValueError

        # NOTE: Important keys here are: 'ar_seg' and "vl_seg"
        annotations = pickle.load(
            open(os.path.join(annotation_path, "ann_trans_data_segments.pickle"), "rb"))

        self.ppg = torch.tensor(physio_trans_data_segments['filt_PPG'])

        self.ppg = (self.ppg - self.ppg.mean(dim=0, keepdim=True)
                    ) / self.ppg.std(dim=0, keepdim=True)

        assert self.ppg.mean(dim=0).mean() < 1e-6, self.ppg.mean(dim=0).sum()
        assert self.ppg.std(dim=0).mean() - 1 < 1e-6, self.ppg.std(dim=0).sum()

        self.valence = torch.tensor(annotations['vl_seg']) - 1
        self.arousal = torch.tensor(annotations['ar_seg']) - 1

        df = []
        for i in range(len(self.ppg)):
            # if (self.ppg[i] == 0.0).all():
            #     continue
            df.append({"ppg": self.ppg[i].numpy(), "val": float(
                self.valence[i]), "aro": float(self.arousal[i]), "quality_idx": i})

        idx_to_keep = physio_trans_data_segments["PPG_quality_idx"]

        df = pd.DataFrame(df)
        old_len = len(df)
        df = df[df["quality_idx"].isin(idx_to_keep)]
        new_len = len(df)
        print(f"Removed {old_len - new_len} bad quality samples")

        self.data = df

        # old_len = len(self.data)
        # self.data = self.remove_bad_quality_samples(self.data)
        # new_len = len(self.data)
        # print(f"Removed {old_len - new_len} bad quality samples")
        # raise ValueError

        self.train_df, self.val_df = train_test_split(
            self.data, test_size=0.2, stratify=self.data[["val", "aro"]], random_state=RANDOM_SEED)
        self.test_df = self.val_df
        # self.val_df, self.test_df = train_test_split(
        #     temp_df, test_size=0.1, stratify=temp_df[["val", "aro"]], random_state=RANDOM_SEED)

        # self.train_df, temp_df = train_test_split(
        #     self.data, test_size=0.1, random_state=RANDOM_SEED)
        # self.val_df, self.test_df = train_test_split(
        #     temp_df, test_size=0.5, random_state=RANDOM_SEED)

        if AUGMENTATION_SIZE > 0:
            self.train_df = GREXTransform(
                self.train_df).augment(n=AUGMENTATION_SIZE)

        if BALANCE_DATASET:
            self.train_df = GREXTransform(self.train_df).balance()

        print(
            f"Valence count TRAIN (after balance): {self.train_df['val'].value_counts()}")
        print(
            f"Arousal count TRAIN (after balance): {self.train_df['aro'].value_counts()}")

        print(
            f"Valence count VAL: {self.val_df['val'].value_counts()}")
        print(
            f"Arousal count VAL: {self.val_df['aro'].value_counts()}")

        self.train_df = extract_ppg_features_from_df(self.train_df)
        self.val_df = extract_ppg_features_from_df(self.val_df)
        self.test_df = extract_ppg_features_from_df(self.test_df)

        print(
            f"Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")

    def remove_bad_quality_samples(self, df):
        quality_segments_path = os.path.join(
            DATA_DIR, "GREX", '6_Results', 'PPG', 'segments', 'Quality')

        bad_df = pd.read_csv(os.path.join(
            quality_segments_path, "PPG_quality_bad_segments.csv"))

        bad_df['idx'] = bad_df['User'].str.split(
            '_').str[-1].str.extract('(\d+)', expand=False)

        # bad_df.to_csv("bad_quality_ppg.csv", index=False)

        bad_df = bad_df.dropna(subset=['idx'])
        # df.to_csv("all_ppg.csv", index=False)

        # Ensure 'idx' is integer type for comparison
        bad_df['idx'] = bad_df['idx'].astype(int)
        df = df[~df['quality_idx'].isin(bad_df['idx'])]

        # df.to_csv("good_quality_ppg.csv", index=False)
        return df

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
