from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import RAVDESS_DF_SPLITTING
from tqdm import tqdm

class RAVDESSDataLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.indices = list(range(self.dataset_size))

    def get_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)
        
def get_train_val_dataloaders(dataset, batch_size):
    train_df, val_df = train_test_split(dataset, test_size=RAVDESS_DF_SPLITTING[0])
    train_loader = tqdm(RAVDESSDataLoader(train_df, batch_size).get_dataloader(), desc="Loading Train Dataloader", leave=False)
    val_loader = tqdm(RAVDESSDataLoader(val_df, batch_size).get_dataloader(), desc="Loading Validation Dataloader", leave=False)
    return train_loader, val_loader, val_df

def get_test_dataloader(dataset, batch_size):
    test_loader = tqdm(RAVDESSDataLoader(dataset, batch_size).get_dataloader(), desc="Loading Testing Dataloader", leave=False)
    return test_loader