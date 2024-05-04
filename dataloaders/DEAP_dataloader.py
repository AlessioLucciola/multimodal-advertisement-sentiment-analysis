
from config import (
    DATA_DIR,
    LENGTH,
    RANDOM_SEED,
    STEP,
    WT
)
import pickle
from shared.constants import CEAP_MEAN, CEAP_STD
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Tuple
import os
import torch
import numpy as np
import pandas as pd
from datasets.DEAP_dataset import DEAPDataset
import json
from utils.ppg_utils import wavelet_transform

class CEAPDataLoader(DataLoader):
    def __init__(self, 
                 batch_size: int,
                 normalize: bool = False):
        data_df = self.load_data()
        print(data_df)
        pass

    def load_data(self) -> pd.DataFrame:
        data_dir = os.path.join(DATA_DIR, "DEAP", "data")
        metadata_dir = os.path.join(DATA_DIR, "DEAP", "metadata")
        ppg_channel = 39
        df = []

        for file in os.listdir(data_dir):
            if not file.endswith(".dat"):
                continue
            abs_path = os.path.join(data_dir, file)  
            print(f"reading file: {abs_path}")
            with open(abs_path, "rb") as f:
                subject = pickle.load(f, encoding='latin1') #resolve the python 2 data problem by encoding : latin1
                for trial_i in range(40):
                    # NOTE: valence is in range [1,9]
                    valence: np.ndarray = subject["labels"][trial_i][0] # 1 is arousal
                    data: np.ndarray = subject["data"][trial_i][ppg_channel]
                    df.append({"ppg": data, "valence": valence})
        return pd.DataFrame(df)

    def get_train_dataloader(self):
        raise NotImplementedError("Train loader not implemented for DEAP")

    def get_val_dataloader(self):
        raise NotImplementedError("Validation loader not implemented for DEAP")

    def get_test_dataloader(self):
        #TODO: implement this
        return 
        dataset = DEAPDataset(self.test_df)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    # Test the dataloader
    dataloader = CEAPDataLoader(batch_size=32)
    # test_loader = dataloader.get_test_dataloader()

    # for i, data in enumerate(test_loader):
    #     print(f"Data size is {len(data)}")
    #     break
