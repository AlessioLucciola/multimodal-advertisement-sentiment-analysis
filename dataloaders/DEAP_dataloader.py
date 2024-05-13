from config import (
    DATA_DIR,
    LENGTH,
    RANDOM_SEED,
    STEP,
)
import pickle
from shared.constants import CEAP_MEAN, CEAP_STD
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List
import os
import torch
import numpy as np
import pandas as pd
from datasets.DEAP_dataset import DEAPDataset
from utils.ppg_utils import wavelet_transform


class DEAPDataLoader(DataLoader):
    def __init__(self,
                 batch_size: int):
        self.batch_size = batch_size
        self.data = self.load_data()
        self.data = self.slice_data(self.data)
        self.data["valence"] = self.data["valence"].apply(lambda x: self.discretize_labels(torch.tensor(x)))

        #scale and normalize according to CEAP std and mean
        print(f"non-normalized data: {self.data}")
        self.data["ppg"] = self.data["ppg"].apply(lambda x: np.array(x))
        cat_data = np.concatenate(self.data["ppg"], axis=0)
        mean, std = cat_data.mean(), cat_data.std()
        print(f"Before DEAP mean + std: {mean, std}")
        self.data["ppg"] = CEAP_MEAN + (self.data["ppg"] - mean) * (CEAP_STD / std)
        # self.data["ppg"] = (self.data["ppg"] - CEAP_MEAN) / CEAP_STD
        print(f"Normalized data: {self.data}")
        cat_data = np.concatenate(self.data["ppg"], axis=0)
        mean, std = cat_data.mean(), cat_data.std()
        print(f"Normalized DEAP mean + std: {mean, std}")
        print(f"normalized data: {self.data}")

        tqdm.pandas()
        self.data["ppg"] = self.data["ppg"].progress_apply(wavelet_transform)
        print(self.data)
        label_counts = self.data['valence'].value_counts()
        print(f"label counts is: {label_counts}")
        
        #TODO: see if this undersampling is needed

        # target_count = label_counts.min()
        # # Sample function to get balanced sample from each group
        # def balanced_sample(group):
        #   return group.sample(target_count, random_state=RANDOM_SEED)
        # # Apply sample function to each group in the DataFrame 
        # self.data = self.data.groupby('valence').apply(balanced_sample)

        label_counts = self.data['valence'].value_counts()
        print(f"label counts after balance is: {label_counts}")

    def load_data(self) -> pd.DataFrame:
        data_dir = os.path.join(DATA_DIR, "DEAP", "data")
        # metadata_dir = os.path.join(DATA_DIR, "DEAP", "metadata")
        ppg_channel = 39
        df = []

        for file in os.listdir(data_dir):
            if not file.endswith(".dat"):
                continue
            abs_path = os.path.join(data_dir, file)
            print(f"reading file: {abs_path}")
            with open(abs_path, "rb") as f:
                # resolve the python 2 data problem by encoding : latin1
                subject = pickle.load(f, encoding='latin1')
                for trial_i in range(40):
                    # NOTE: valence is in range [1,9]
                    valence: np.ndarray = subject["labels"][trial_i][0] #index 0 is valence, 1 arousal
                    data: np.ndarray = subject["data"][trial_i][ppg_channel]
                    df.append({"ppg": data, "valence": valence})
        return pd.DataFrame(df)

    def slice_data(self, df, length=LENGTH, step=STEP) -> pd.DataFrame:
        if length > 3000:
            raise ValueError(f"Length cannot be greater than original length")
        new_df = []
        for _, row in df.iterrows():
            ppg, label = row["ppg"], row["valence"]
            for i in range(0, len(ppg) - length + 1, step):
                ppg_segment = ppg[i:i+length]
                new_row = {
                        "ppg": ppg_segment ,
                        "valence": label,
                        "segment": i}
                new_df.append(new_row)
        new_df = pd.DataFrame(new_df)
        return new_df
    
    def discretize_labels(self, valence: torch.Tensor) -> List[float]:
        self.labels = torch.full_like(valence, -1)
        self.labels[(valence >= 1) & (valence <= 4) ] = 0 
        self.labels[(valence > 4) & (valence <= 6) ] = 1
        self.labels[(valence > 6) & (valence <= 9) ] = 2
        return self.labels.tolist()

    def get_train_dataloader(self):
        raise NotImplementedError("Train loader not implemented for DEAP")

    def get_val_dataloader(self):
        raise NotImplementedError("Validation loader not implemented for DEAP")

    def get_test_dataloader(self):
        dataset = DEAPDataset(self.data)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


if __name__ == "__main__":
    # Test the dataloader
    dataloader = DEAPDataLoader(batch_size=32)
    # test_loader = dataloader.get_test_dataloader()

    # for i, data in enumerate(test_loader):
    #     print(f"Data size is {len(data)}")
    #     break
