from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import RAVDESS_DF_SPLITTING, RANDOM_SEED
import pandas as pd

from datasets.RAVDESS_dataset import RAVDESSCustomDataset

class RAVDESSDataLoader(DataLoader):
    def __init__(self,
                 csv_file: str,
                 batch_size: int,
                 audio_files_dir: str,
                 seed: int = RANDOM_SEED):
        self.batch_size = batch_size
        self.data = pd.read_csv(csv_file)
        self.dataset_size = len(self.data)
        self.audio_files_dir = audio_files_dir
        self.seed = seed

        self.train_df, temp_df = train_test_split(self.data, test_size=RAVDESS_DF_SPLITTING[0], random_state=self.seed)
        self.val_df, self.test_df = train_test_split(temp_df, test_size=RAVDESS_DF_SPLITTING[1], random_state=self.seed)

    def get_train_dataloader(self):
        train_dataset = RAVDESSCustomDataset(data=self.train_df, files_dir=self.audio_files_dir, isTrainDataset=True)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_val_dataloader(self):
        val_dataset = RAVDESSCustomDataset(data=self.val_df, files_dir=self.audio_files_dir, isTrainDataset=False)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def get_test_dataloader(self):
        test_dataset = RAVDESSCustomDataset(data=self.test_df, files_dir=self.audio_files_dir, isTrainDataset=False)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
