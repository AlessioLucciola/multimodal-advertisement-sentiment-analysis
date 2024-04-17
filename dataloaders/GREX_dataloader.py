from config import AUGMENTATION_SIZE, BALANCE_DATASET, DATA_DIR, LENGTH, LOAD_DF, MODEL_NAME, RANDOM_SEED, SAVE_DF, STEP, WAVELET_STEP, ADD_NOISE
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets.GREX_dataset import GREXDataset
from scipy.interpolate import CubicSpline
from utils.ppg_utils import wavelet_transform


class GREXTransform:
    def __init__(self, df):
        self.df = df
        self.transformations = set()

    def jitter(self, data):
        sigma_min, sigma_max = 0.01, 0.3
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
        shift = np.random.randint(low=-LENGTH, high=LENGTH)
        return np.roll(data, shift)
    
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

    def apply_jitter(self):
        new_df = []
        for i, row in self.df.iterrows():
            ppg, val, aro = row["ppg"], row["val"], row["aro"]
            new_item = {"ppg": self.jitter(ppg), "val": val, "aro": aro}
            print(f"type of jittered ppg is: {type(new_item['ppg'])}")
            new_df.append(new_item)
        return pd.DataFrame(new_df)

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


        # NOTE: Important keys here are: 'ar_seg' and "vl_seg"
        annotations = pickle.load(
            open(os.path.join(annotation_path, "ann_trans_data_segments.pickle"), "rb"))

        self.ppg = torch.tensor(physio_trans_data_segments['filt_PPG'])

        # NOTE: Use valence in [0,4] and arousal in [0,4]
        self.valence = torch.tensor(annotations['vl_seg']) - 1
        self.arousal = torch.tensor(annotations['ar_seg']) - 1
        # self.uncertain = annotations['unc_seg']

        # OR use bad, neutral and good mood
        self.labels = self.set_labels(self.valence, self.arousal)

        # TODO: just for debug because I'm lazy
        self.valence = self.labels
        self.arousal = self.labels

        df = []
        for i in range(len(self.ppg)):
            if (self.ppg[i] == 0.0).all():
                continue

            #TODO: change "val" here to "label", and also in the train and dataset
            df.append({"ppg": self.ppg[i].numpy(), "val": int(
                self.valence[i]), "aro": int(self.arousal[i]), "quality_idx": i})

        idx_to_keep = physio_trans_data_segments["PPG_quality_idx"]

        df = pd.DataFrame(df)
        old_len = len(df)
        df = df[df["quality_idx"].isin(idx_to_keep)]
        new_len = len(df)
        print(f"Removed {old_len - new_len} bad quality samples")

        self.data = df

        # Apply standardization
        ppg_mean = np.stack(self.data["ppg"].to_numpy(), axis=0).mean()
        ppg_std = np.stack(self.data["ppg"].to_numpy(), axis=0).std()
        self.data["ppg"] = self.data["ppg"].apply(
            lambda x: (x - ppg_mean) / ppg_std)


        # if not LOAD_DF:
        #     tqdm.pandas()
        #     self.data["ppg_spatial_features"] = self.data["ppg"].progress_apply(
        #         wavelet_transform)

        # if SAVE_DF:
        #     self.save_wavelets(self.data, f"data_{LENGTH}_{WAVELET_STEP}")

        # if LOAD_DF:
        #     print("Loading dataframes...")
        #     self.data = self.load_wavelets(self.data, f"data_{LENGTH}_{WAVELET_STEP}")


        self.data, self.test_df = train_test_split(self.data,
                                                   test_size=0.1,
                                                   stratify=self.data["val"],
                                                   random_state=RANDOM_SEED)

        # self.data = self.slice_data(self.data)

        self.train_df, self.val_df = train_test_split(
                self.data, test_size=0.3,
                stratify=self.data["val"],
                random_state=RANDOM_SEED)

        self.train_df = self.slice_data(self.train_df)
        self.val_df = self.slice_data(self.val_df)

        if ADD_NOISE:
            self.train_df = GREXTransform(self.train_df).apply_jitter()
            # self.val_df = GREXTransform(self.val_df).apply_jitter()

        if not LOAD_DF:
            tqdm.pandas()
            self.train_df["ppg_spatial_features"] = self.train_df["ppg"].progress_apply(
                wavelet_transform)
            self.val_df["ppg_spatial_features"] = self.val_df["ppg"].progress_apply(
                wavelet_transform)
              # self.test_df["ppg_spatial_features"] = self.test_df["ppg"].progress_apply(
              #    wavelet_transform)

        if SAVE_DF:
            self.save_wavelets(self.train_df, f"train_df_{LENGTH}_{WAVELET_STEP}")
            self.save_wavelets(self.val_df, f"val_df_{LENGTH}_{WAVELET_STEP}")

        if LOAD_DF:
            print("Loading dataframes...")
            self.train_df = self.load_wavelets(self.train_df, f"train_df_{LENGTH}_{WAVELET_STEP}")
            self.val_df = self.load_wavelets(self.val_df, f"val_df_{LENGTH}_{WAVELET_STEP}")

        # ppg_mean = np.stack(self.train_df["ppg_spatial_features"].to_numpy(), axis=0).mean(axis=0)
        # ppg_std = np.stack(self.train_df["ppg_spatial_features"].to_numpy(), axis=0).std(axis=0)
        # self.train_df["ppg_spatial_features"] = self.train_df["ppg_spatial_features"].apply(
        #     lambda x: (x - ppg_mean) / ppg_std)
        # ppg_mean = np.stack(self.val_df["ppg_spatial_features"].to_numpy(), axis=0).mean(axis=0)
        # ppg_std = np.stack(self.val_df["ppg_spatial_features"].to_numpy(), axis=0).std(axis=0)
        # self.val_df["ppg_spatial_features"] = self.val_df["ppg_spatial_features"].apply(
        #     lambda x: (x - ppg_mean) / ppg_std)

        if BALANCE_DATASET:
            self.train_df = self.undersample(self.train_df)
            self.val_df = self.undersample(self.val_df)
            self.test_df = self.undersample(self.test_df)

        if AUGMENTATION_SIZE > 0:
            raise ValueError("augmentation is deprecated")
            self.train_df = GREXTransform(
                self.train_df).augment(n=AUGMENTATION_SIZE)


        # if BALANCE_DATASET:
        #     raise ValueError("augmentation is deprecated")
        #     self.train_df = GREXTransform(self.train_df).balance()

        print(
            f"Train group size: \n{self.train_df.groupby(['val']).size()}")
        print(
            f"Validation group size: \n {self.val_df.groupby(['val']).size()}")

        print(
            f"Test group size: \n {self.test_df.groupby(['val']).size()}")

        print(
            f"Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")

    def save_wavelets(self, df: pd.DataFrame, name: str):
        ppg_spatial_features = np.stack(df["ppg_spatial_features"].to_numpy(), axis=0)
        np.save(f"ppg_spatial_features_{name}.npy", ppg_spatial_features)

    def load_wavelets(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        path = f"ppg_spatial_features_{name}.npy"
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} cannot be found")
        ppg_spatial_features = np.load(path)
        assert len(df) == len(ppg_spatial_features)
        loaded_df = []
        for i, row in df.iterrows():
            if i >= len(ppg_spatial_features):
                # TODO: see why the i goes until 1235 if len(self.data) is 1213
                print(f"Skipping {i}")
                continue
            ppg_spatial_feature = ppg_spatial_features[i]
            new_row = {"ppg": row["ppg"], "val": row["val"],
                       "aro": row["aro"], "ppg_spatial_features": ppg_spatial_feature}
            loaded_df.append(new_row)
        return pd.DataFrame(loaded_df)

    def undersample(self, df):
        # Group by 'aro' and 'val' columns and get the size of each group
        group_sizes = df.groupby(['val']).size()
        # Get the minimum group size
        min_size = group_sizes.min()
        # Function to apply to each group

        def undersample_group(group):
            return group.sample(min_size)
        # Apply the function to each group
        df_undersampled = df.groupby(['val']).apply(undersample_group)
        # Reset the index (groupby and apply result in a multi-index)
        df_undersampled.reset_index(drop=True, inplace=True)
        return df_undersampled

    def slice_data(self, df, length=LENGTH, step=STEP):
        """
        Taken a dataframe that contains data with columns "ppg", "val", "aro",
        this function outputs a new df where the data is sliced into segments of
        length `length` with a sliding window step of `step`.
        """
        if length > 2000:
            raise ValueError(
                "Length cannot be greater than original length 2000")
        new_df = []
        for _, row in df.iterrows():
            if "ppg_spatial_features" in row:
                ppg, val, aro, wavelet = row["ppg"], row["val"], row["aro"], row["ppg_spatial_features"]
            else:
                ppg, val, aro = row["ppg"], row["val"], row["aro"]
            for i in range(0, len(ppg) - length + 1, step):
                assert ppg[i:i +
                           length].shape[0] == length, f"Shape is not consistent: {ppg[i:i + length].shape} != {length}"
                segment = ppg[i:i + length]
                if "ppg_spatial_features" in row:
                    segment_wavelet = wavelet[:, i:i + length]
                    print(f"segment wavelet shape is {segment_wavelet.shape}")
                    new_row = {"ppg": segment, "val": val, "aro": aro,
                               "ppg_spatial_features": segment_wavelet}
                else:
                    new_row = {"ppg": segment, "val": val, "aro": aro}
                new_df.append(new_row)

        new_df = pd.DataFrame(new_df)

        # # TODO: see if it makes sense to standardize here
        # ppg_mean = np.stack(new_df["ppg"].to_numpy(), axis=0).mean()
        # ppg_std = np.stack(new_df["ppg"].to_numpy(), axis=0).std()
        # new_df["ppg"] = new_df["ppg"].apply(
        #     lambda x: (x - ppg_mean) / ppg_std)

        # # TODO: see if it makes sense to take the mean on the 0th axis or should I do something different
        # features_mean = np.stack(
        #     new_df["ppg_spatial_features"].to_numpy(), axis=0).mean(axis=0)
        # features_std = np.stack(
        #     new_df["ppg_spatial_features"].to_numpy(), axis=0).std(axis=0)
        # new_df["ppg_spatial_features"] = new_df["ppg_spatial_features"].apply(
        #     lambda x: (x - features_mean) / features_std)

        return new_df

    def set_labels(self, valence: torch.Tensor, arousal: torch.Tensor):
        # bad mood: valence == 0,1; neutral: valence == 2, good: valence = 3,4
        self.labels = torch.zeros_like(valence)
        self.labels[(valence == 0) | (valence == 1)] = 0
        self.labels[valence == 2] = 1
        self.labels[(valence == 3) | (valence == 4)] = 2
        print(f"Labels are {self.labels}")
        return self.labels

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
        print(f"Data size is {len(data)}")
        print(f"data is {data}")
        break