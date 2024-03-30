from datasets.FER_dataset import FERDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import RANDOM_SEED
import pandas as pd

class FERDataloader:
    def __init__(self, 
                 csv_file: str, 
                 batch_size: int, 
                 val_size: float,
                 seed: int = RANDOM_SEED,
                 limit: int = None,
                 balance_dataset: bool = True):
        self.batch_size = batch_size
        self.data = pd.read_csv(csv_file)
        self.val_size = val_size
        self.seed = seed
        self.limit = limit
        self.balance_dataset = balance_dataset


        if self.limit is not None:
            if self.limit <= 0 or self.limit > 1:
                return ValueError("Limit must be a float in the range (0, 1] or None")
            else:
                self.data = self.data.sample(frac=self.limit, random_state=self.seed)
                print(f"--Dataloader-- Limit parameter set to {self.limit}. Using {self.limit*100}% of the dataset.")

    def get_train_val_dataloader(self):
        train_dataset = FERDataset(data=self.data, is_train_dataset=True, transform=None, balance_dataset=self.balance_dataset)
        val_len = int(self.val_size*len(train_dataset))
        train_ds, val_ds = random_split(train_dataset, [len(train_dataset)-val_len, val_len])
        print(f"--Dataset-- Training dataset size: {len(train_ds)}")
        print(f"--Dataset-- Validation dataset size: {len(val_ds)}")
        return DataLoader(train_ds, batch_size=self.batch_size, shuffle=True), DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
    
    def get_test_dataloader(self):
        test_dataset = FERDataset(data=self.data, is_train_dataset=False, transform=None, balance_dataset=self.balance_dataset)
        print(f"--Dataset-- Test dataset size: {len(test_dataset)}")
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)