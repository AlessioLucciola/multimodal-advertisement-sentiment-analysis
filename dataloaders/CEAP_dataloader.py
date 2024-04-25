
from config import (
    AUGMENTATION_SIZE,
    BALANCE_DATASET,
    DATA_DIR,
    LENGTH,
    LOAD_DF,
    RANDOM_SEED,
    SAVE_DF,
    STEP,
    WAVELET_STEP,
    ADD_NOISE,
)
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets.CEAP_dataset import CEAPDataset
import json
from typing import Literal
from enum import Enum


class CEAPDataLoader(DataLoader):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        _type = "Frame" # Raw, Frame or Transformed
        data_path = os.path.join(DATA_DIR, "CEAP", "5_PhysioData", _type)
        annotation_path = os.path.join(DATA_DIR, "CEAP", "3_AnnotationData", _type)

        data_df = []  
        for file in os.listdir(data_path):
            df_item = {}
            if not file.endswith("json"):
                continue
            with open(os.path.join(data_path, file), "r") as f:
                parsed_json = json.load(f)
            participant_id = parsed_json[f"Physio_{_type}Data"][0]["ParticipantID"]
            video_list = parsed_json[f"Physio_{_type}Data"][0][f"Video_Physio_{_type}Data"]
            for video_info in video_list:
                # NOTE: the PPG can have a length of 1800 or 1500
                video_id = video_info["VideoID"]
                ppg_list = [item["BVP"] for item in video_info[f"BVP_{_type}Data"]]
                #TODO: see how to also put the labels here, without modifying the df later
                df_item  = {"participant_id": participant_id, 
                            "video_id": video_id, 
                            "ppg": ppg_list}
                data_df.append(df_item)
            print(data_df)



        #TODO: continue

    def get_train_dataloader(self):
        dataset = CEAPDataset(self.train_df)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def get_val_dataloader(self):
        dataset = CEAPDataset(self.val_df)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def get_test_dataloader(self):
        dataset = CEAPDataset(self.test_df)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    # Test the dataloader
    dataloader = CEAPDataLoader(batch_size=32)
    raise NotImplementedError()
    train_loader = dataloader.get_train_dataloader()
    val_loader = dataloader.get_val_dataloader()
    test_loader = dataloader.get_test_dataloader()

    for i, data in enumerate(train_loader):
        print(f"Data size is {len(data)}")
        print(f"data is GREXDataLoader {data}")
        break
