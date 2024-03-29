from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import DF_SPLITTING, RANDOM_SEED
import pandas as pd
from datasets.voice_custom_dataset import RAVDESSCustomDataset

class RAVDESSDataLoader(DataLoader):
    def __init__(self,
                 csv_file: str,
                 batch_size: int,
                 audio_files_dir: str,
                 seed: int = RANDOM_SEED,
                 limit: int = None,
                 balance_dataset: bool = True,
                 preload_audio_files: bool = True,
                 scale_audio_files: bool = True
                 ):
        self.batch_size = batch_size
        self.data = pd.read_csv(csv_file)
        self.dataset_size = len(self.data)
        self.audio_files_dir = audio_files_dir
        self.seed = seed
        self.limit = limit
        self.balance_dataset = balance_dataset
        self.preload_audio_files = preload_audio_files
        self.scale_audio_files = scale_audio_files

        if self.limit is not None:
            if self.limit <= 0 or self.limit > 1:
                return ValueError("Limit must be a float in the range (0, 1] or None")
            else:
                self.data = self.data.sample(frac=self.limit, random_state=self.seed)
                print(f"--Dataloader-- Limit parameter set to {self.limit}. Using {self.limit*100}% of the dataset.")

        self.train_df, temp_df = train_test_split(self.data, test_size=DF_SPLITTING[0], random_state=self.seed)
        self.val_df, self.test_df = train_test_split(temp_df, test_size=DF_SPLITTING[1], random_state=self.seed)

    def get_train_dataloader(self):
        train_dataset = RAVDESSCustomDataset(data=self.train_df, files_dir=self.audio_files_dir, is_train_dataset=True, balance_dataset=self.balance_dataset, preload_audio_files=self.preload_audio_files, scale_audio_files=self.scale_audio_files)
        self.scaler = train_dataset.scaler
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_val_dataloader(self):
        val_dataset = RAVDESSCustomDataset(data=self.val_df, files_dir=self.audio_files_dir, is_train_dataset=False, preload_audio_files=self.preload_audio_files, scale_audio_files=self.scale_audio_files, scaler=self.scaler)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def get_test_dataloader(self, scaler):
        test_dataset = RAVDESSCustomDataset(data=self.test_df, files_dir=self.audio_files_dir, is_train_dataset=False, preload_audio_files=self.preload_audio_files, scale_audio_files=self.scale_audio_files, scaler=scaler)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
