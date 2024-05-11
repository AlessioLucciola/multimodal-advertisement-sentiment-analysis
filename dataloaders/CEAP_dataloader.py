from config import (
    DATA_DIR,
    LENGTH,
    RANDOM_SEED,
    STEP,
    WT
)
from shared.constants import CEAP_MEAN, CEAP_STD
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Tuple
import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from datasets.CEAP_dataset import CEAPDataset
import json
from utils.ppg_utils import wavelet_transform

class CEAPDataLoader(DataLoader):
    def __init__(self, 
                 batch_size: int,
                 normalize: bool = False):
        self.batch_size = batch_size
        print("Loading data...")
        data_type = "Frame"
        self.data = self.load_data(data_type=data_type, 
                                   annotation_type=data_type,
                                   data_path=os.path.join(DATA_DIR, "CEAP", "5_PhysioData", data_type),
                                   annotation_path=os.path.join(DATA_DIR, "CEAP", "3_AnnotationData", data_type))
        

        print("Discretizing labels...")
        self.data["valence"] = self.data["valence"].apply(lambda x: self.discretize_labels(torch.tensor(x)))

        mean, std = self.get_stats() 
        print(f"Stats of raw data are: \n Mean: {mean} \n Std: {std}")
        
        # Convert lists to numpy arrays
        self.data["ppg"] = self.data["ppg"].apply(lambda x: np.array(x))
        if normalize:
            raise NotImplementedError("There is no need to normalize")
            cat_data = np.concatenate(self.data["ppg"], axis=0)
            mean, std = cat_data.mean(), cat_data.std()
            print(f"Before CEAP mean + std: {mean, std}")
            scaling_factor = (CEAP_STD / std) + (CEAP_MEAN - mean)
            self.data["ppg"] = self.data["ppg"] * scaling_factor
            self.data["ppg"] = (self.data["ppg"] - CEAP_MEAN) / CEAP_STD
            print(f"Normalized data: {self.data}")
            cat_data = np.concatenate(self.data["ppg"], axis=0)
            mean, std = cat_data.mean(), cat_data.std()
            print(f"Normalized CEAP mean + std: {mean, std}")
        

        print("Splitting data...")
        self.train_df, self.val_df, self.test_df = self.split_data()

        print("Slicing data...")
        self.train_df = self.slice_data(self.train_df)
        self.val_df = self.slice_data(self.val_df)
        self.test_df = self.slice_data(self.test_df)
        
        if WT:
            print(f"Performing wavelet transform...")
            tqdm.pandas()
            self.train_df["ppg"] = self.train_df["ppg"].progress_apply(wavelet_transform)
            self.val_df["ppg"] = self.val_df["ppg"].progress_apply(wavelet_transform)
            self.test_df["ppg"] = self.test_df["ppg"].progress_apply(wavelet_transform)
        else:
            print("Skipped wavelet transform")

        print("Finished!")

        print(f"Train_df length: {len(self.train_df)}")
        print(f"Val_df length: {len(self.val_df)}")
        print(f"Test_df length: {len(self.test_df)}")


    def load_data(self,
                  data_type: str, 
                  annotation_type: str,
                  data_path: str,
                  annotation_path: str,
                  load_labels: bool = True) -> pd.DataFrame:
        data_df = []  
        annotation_df = []
        # Take data
        for file in os.listdir(data_path):
            df_item = {}
            if not file.endswith("json"):
                continue
            with open(os.path.join(data_path, file), "r") as f:
                parsed_json = json.load(f)
            participant_id = parsed_json[f"Physio_{data_type}Data"][0]["ParticipantID"].replace("P", "")
            video_list = parsed_json[f"Physio_{data_type}Data"][0][f"Video_Physio_{data_type}Data"]
            for video_info in video_list:
                video_id = video_info["VideoID"]
                ppg_list = [item["BVP"] for item in video_info[f"BVP_{data_type}Data"]]
                df_item  = {"participant_id": participant_id, 
                            "video_id": video_id, 
                            "ppg": ppg_list}
                data_df.append(df_item)

        if not load_labels:
            return pd.DataFrame(data_df)

        # Take annotations (they are sampled at 10Hz)
        for file in os.listdir(annotation_path):
            df_item = {}
            if not file.endswith("json"):
                continue
            with open(os.path.join(annotation_path, file), "r") as f:
                parsed_json = json.load(f)
            participant_id = parsed_json[f"ContinuousAnnotation_{annotation_type}Data"][0]["ParticipantID"]
            video_list = parsed_json[f"ContinuousAnnotation_{annotation_type}Data"][0][f"Video_Annotation_{annotation_type}Data"]
            for video_info in video_list:
                video_id = video_info["VideoID"]
                valence_list = [item["Valence"] for item in video_info[f"TimeStamp_Valence_Arousal"]]
                df_item  = {"participant_id": participant_id, 
                            "video_id": video_id, 
                            "valence": valence_list}
                annotation_df.append(df_item)

        # Combine annotations and data into a single df
        df = []
        for data_item in data_df:
            for annotation_item in annotation_df:
                if data_item["participant_id"] != annotation_item["participant_id"] or \
                        data_item["video_id"] != annotation_item["video_id"]:
                            continue
                df_item = {
                        "participant_id": data_item["participant_id"],
                        "video_id": data_item["video_id"],
                        "ppg": data_item["ppg"],
                        "valence": annotation_item["valence"],
                        }
                df.append(df_item)

        df = pd.DataFrame(df)
        return df
    
    def discretize_labels(self, valence: torch.Tensor) -> List[float]:
        self.labels = torch.full_like(valence, -1)
        self.labels[(valence >= 1) & (valence <= 4) ] = 0 
        self.labels[(valence > 4) & (valence <= 6) ] = 1
        self.labels[(valence > 6) & (valence <= 9) ] = 2
        return self.labels.tolist()

    def split_data(self):
        # train_df, temp_df = train_test_split(self.data, test_size=0.2, random_state=RANDOM_SEED)
        # val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED)
        # return train_df, val_df, test_df
        df = self.data

        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2
        if train_ratio + val_ratio + test_ratio != 1:
            raise ValueError("Ratios must sum to 1. Please adjust the values.")

        pid_groups = df['participant_id'].tolist()
        X = df['ppg'].tolist()
        Y = df['valence'].tolist()
        sss = GroupShuffleSplit(n_splits=1, test_size=val_ratio + test_ratio, random_state=RANDOM_SEED)

        # Split the dataframe based on pid groups
        for train_index, test_index in sss.split(X, Y, pid_groups):  # Splitting based on labels maintains class balance
            train_df = df.iloc[train_index]
            remaining = df.iloc[test_index]

            # Further split remaining data into validation and test sets (optional)
            sss_inner = GroupShuffleSplit(n_splits=1, test_size=test_ratio / (val_ratio + test_ratio), random_state=RANDOM_SEED)
            X_remaining, Y_remaining = remaining["ppg"].tolist(), remaining["valence"].tolist()    
            pid_groups_remaining = remaining['participant_id'].tolist()
            val_index, test_index = next(sss_inner.split(X_remaining, Y_remaining, pid_groups_remaining))
            val_df = remaining.iloc[val_index]
            test_df = remaining.iloc[test_index]

        train_pids = set(train_df["participant_id"].unique())
        val_pids = set(val_df["participant_id"].unique())
        test_pids = set(test_df["participant_id"].unique())
        
        # print("train pids ", train_pids)
        # print("val pids ", val_pids)
        # print("test pids ",test_pids )

        assert len(train_pids & val_pids) == 0
        assert len(train_pids & test_pids) == 0
        assert len(val_pids & test_pids) == 0
         
        return train_df, val_df, test_df

        

    def slice_data(self, df, length=LENGTH, step=STEP) -> pd.DataFrame:
        if length > 1500:
            raise ValueError(f"Length cannot be greater than original length of 1500")
        new_df = []
        for _, row in df.iterrows():
            pid, vid, ppg, labels = row["participant_id"], row["video_id"], row["ppg"], row["valence"]
            for i in range(0, len(ppg) - length + 1, step):
                ppg_segment = ppg[i:i+length]
                label_segment = labels[i:i+length]
                new_row = {
                        "participant_id": pid,
                        "video_id": vid,
                        "ppg": ppg_segment ,
                        "valence": label_segment}
                new_df.append(new_row)
        new_df = pd.DataFrame(new_df)
        return new_df

    def get_stats(self) -> Tuple[float, float]:
        data_type = "Frame"
        raw_data = self.load_data(data_type=data_type, 
                                  annotation_type=data_type,
                                  data_path=os.path.join(DATA_DIR, "CEAP", "5_PhysioData", data_type),
                                  annotation_path=os.path.join(DATA_DIR, "CEAP", "3_AnnotationData", data_type),
                                  load_labels=False)
        
        ppg_mean = np.concatenate(raw_data["ppg"].to_numpy(), axis=0).mean()
        ppg_std = np.concatenate(raw_data["ppg"].to_numpy(), axis=0).std()
        return ppg_mean, ppg_std

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
    dataloader = CEAPDataLoader(batch_size=32, normalize=True)
    train_loader = dataloader.get_train_dataloader()
    val_loader = dataloader.get_val_dataloader()
    test_loader = dataloader.get_test_dataloader()

    for i, data in enumerate(train_loader):
        print(f"Data size is {len(data)}")
        # print(f"data is GREXDataLoader {data}")
        break
