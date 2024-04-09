from datasets.video_custom_dataset import video_custom_dataset
from torch.utils.data import DataLoader
from config import RANDOM_SEED, DF_SPLITTING
from sklearn.model_selection import train_test_split
import pandas as pd

class video_custom_dataloader:
    def __init__(self, 
                 csv_file: str, 
                 batch_size: int, 
                 seed: int = RANDOM_SEED,
                 limit: int = None,
                 apply_transformations: bool = True,
                 balance_dataset: bool = True,
                 use_default_split: bool = True
                 ):
        self.batch_size = batch_size
        self.data = pd.read_csv(csv_file)
        self.seed = seed
        self.limit = limit
        self.apply_transformations = apply_transformations
        self.balance_dataset = balance_dataset
        self.use_default_split = use_default_split

        if self.limit is not None:
            if self.limit <= 0 or self.limit > 1:
                return ValueError("Limit must be a float in the range (0, 1] or None")
            else:
                self.data = self.data.sample(frac=self.limit, random_state=self.seed)
                print(f"--Dataloader-- Limit parameter set to {self.limit}. Using {self.limit*100}% of the dataset.")
        
        if self.use_default_split:
            train_val_df = self.data.loc[self.data.Usage.isin(['Training', 'PublicTest'])]
            self.train_df, self.val_df = train_test_split(train_val_df, test_size=DF_SPLITTING[0], random_state=self.seed)
            self.test_df = self.data.loc[self.data.Usage.isin(['PrivateTest'])]

            # Remove unnecessary columns
            self.train_df = self.train_df.drop(['Usage'], axis=1)
            self.val_df = self.val_df.drop(['Usage'], axis=1)
            self.test_df = self.test_df.drop(['Usage'], axis=1)
        else:
            self.data = self.data.drop(['Usage'], axis=1) # Remove unnecessary columns

            self.train_df, temp_df = train_test_split(self.data, test_size=DF_SPLITTING[0], random_state=self.seed)
            self.val_df, self.test_df = train_test_split(temp_df, test_size=DF_SPLITTING[1], random_state=self.seed)  
            
    def get_train_dataloader(self):
        train_dataset = video_custom_dataset(data=self.train_df, is_train_dataset=True, apply_transformations=self.apply_transformations, balance_dataset=self.balance_dataset)
        print(f"--Dataset-- Training dataset size: {len(train_dataset)}")
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_val_dataloader(self):
        val_dataset = video_custom_dataset(data=self.val_df, is_train_dataset=False, apply_transformations=self.apply_transformations, balance_dataset=self.balance_dataset)
        print(f"--Dataset-- Validation dataset size: {len(val_dataset)}")
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    def get_test_dataloader(self):
        test_dataset = video_custom_dataset(data=self.test_df, is_train_dataset=False, apply_transformations=self.apply_transformations, balance_dataset=self.balance_dataset)
        print(f"--Dataset-- Test dataset size: {len(test_dataset)}")
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)