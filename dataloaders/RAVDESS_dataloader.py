from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import RAVDESS_DF_SPLITTING, RAVDESS_CSV, RAVDESS_FILES
from datasets.RAVDESS_dataset import RAVDESSCustomDataset

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
    train_loader = RAVDESSDataLoader(train_df, batch_size).get_dataloader()
    val_loader = RAVDESSDataLoader(val_df, batch_size).get_dataloader()
    return train_loader, val_loader, val_df

def get_test_dataloader(dataset, batch_size):
    test_loader = RAVDESSDataLoader(dataset, batch_size).get_dataloader()
    return test_loader