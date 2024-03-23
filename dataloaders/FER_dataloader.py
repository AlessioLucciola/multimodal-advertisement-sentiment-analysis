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
                 limit: int = None):
        self.batch_size = batch_size
        self.data = pd.read_csv(csv_file)
        self.val_size = val_size
        self.seed = seed
        self.limit = limit

        if self.limit is not None:
            if self.limit <= 0 or self.limit > 1:
                return ValueError("Limit must be a float in the range (0, 1] or None")
            else:
                self.data = self.data.sample(frac=self.limit, random_state=self.seed)
                print(f"--Dataloader-- Limit parameter set to {self.limit}. Using {self.limit*100}% of the dataset.")

    def get_train_val_dataloader(self):
        train_dataset = FERDataset(data=self.data, split='train', transform=None)
        val_len = int(self.val_size*len(train_dataset))
        train_ds, val_ds = random_split(train_dataset, [len(train_dataset)-val_len, val_len])
        return DataLoader(train_ds, batch_size=self.batch_size, shuffle=True), DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
    
    def get_test_dataloader(self):
        test_dataset = FERDataset(data=self.data, split='test', transform=None)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

# def FERDataloader(data_dir, val_size=0.2, batch_size=32, shuffle=True, transformations=None): # TODO: change into a class
#     '''This is function to load the dataset and returns the dataloaders
#         Args:
#             data_dir: path to the dataset
#             batch_size: batch size for the dataloader
#             num_workers: number of workers for the dataloader
#             shuffle: whether to shuffle the dataset or not
#         Returns:
#             train_loader: dataloader for the training set
#             val_loader: dataloader for the validation set
#             test_loader: dataloader for the test set
#     '''
#     train_dataset = FERDataset(
#         data_dir=data_dir, split='train', transform=transformations)
#     test_dataset = FERDataset(
#         data_dir=data_dir, split='test', transform=transformations)


#     val_len = int(val_size*len(train_dataset))
#     train_ds, val_ds = random_split(
#         train_dataset, [len(train_dataset)-val_len, val_len])

#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle)
#     test_loader = DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=shuffle)

#     return train_loader, val_loader, test_loader
