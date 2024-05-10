from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import RANDOM_SEED
import pandas as pd
from datasets.fer_custom_dataset import fer_custom_dataset
from shared.constants import fer_mapping_to_positive_negative

class fer_custom_dataloader(DataLoader):
    def __init__(self,
                 csv_frames_files: str,
                 batch_size: int,
                 frames_dir: str,
                 seed: int = RANDOM_SEED,
                 limit: int = None,
                 use_positive_negative_labels: bool = True,
                 preload_frames: bool = True,
                 apply_transformations: bool = True,
                 balance_dataset: bool = True,
                 normalize: bool = True,
                 ):
        self.batch_size = batch_size
        self.csv_frames_files = pd.read_csv(csv_frames_files)
        self.frames_dir = frames_dir
        self.seed = seed
        self.limit = limit
        self.use_positive_negative_labels = use_positive_negative_labels
        self.preload_frames = preload_frames
        self.apply_transformations = apply_transformations
        self.balance_dataset = balance_dataset
        self.normalize = normalize

        if self.limit is not None:
            if self.limit <= 0 or self.limit > 1:
                return ValueError("Limit must be a float in the range (0, 1] or None")
            else:
                self.data = self.data.sample(frac=self.limit, random_state=self.seed)
                print(f"--Dataloader-- Limit parameter set to {self.limit}. Using {self.limit*100}% of the dataset.")

        # Split the dataset into train, validation and test using the Usage column
        self.train_df = self.csv_frames_files[self.csv_frames_files['Usage'] == 'Training']
        self.val_df = self.csv_frames_files[self.csv_frames_files['Usage'] == 'PublicTest']
        self.test_df = self.csv_frames_files[self.csv_frames_files['Usage'] == 'PrivateTest']

        # Map the emotions to positive/negative labels
        if self.use_positive_negative_labels:
            print("--Dataloader-- Using positive/negative labels mapping.")
            self.train_df.loc[:, 'emotion'] = self.train_df['emotion'].astype(int).map(fer_mapping_to_positive_negative)
            self.val_df.loc[:, 'emotion'] = self.val_df['emotion'].astype(int).map(fer_mapping_to_positive_negative)
            self.test_df.loc[:, 'emotion'] = self.test_df['emotion'].astype(int).map(fer_mapping_to_positive_negative)

        # Print unique values in the column 'emotion'
        print(f"--- Train emotions: {self.train_df['emotion'].unique()} \n--- Validation emotions: {self.val_df['emotion'].unique()} \n--- Test emotions: {self.test_df['emotion'].unique()}")

    def get_train_dataloader(self):
        train_dataset = fer_custom_dataset(data=self.train_df, 
                                              files_dir=self.frames_dir, 
                                              is_train_dataset=True, 
                                              preload_frames=self.preload_frames, 
                                              balance_dataset=self.balance_dataset, 
                                              apply_transformations=self.apply_transformations, 
                                              normalize=self.normalize
                                              )
        print(f"--Dataset-- Train dataset size: {train_dataset.__len__()}")
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_val_dataloader(self):
        val_dataset = fer_custom_dataset(data=self.val_df, 
                                            files_dir=self.frames_dir, 
                                            is_train_dataset=False, 
                                            preload_frames=self.preload_frames, 
                                            balance_dataset=self.balance_dataset, 
                                            apply_transformations=self.apply_transformations, 
                                            normalize=self.normalize
                                            )
        print(f"--Dataset-- Validation dataset size: {val_dataset.__len__()}")
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def get_test_dataloader(self):
        test_dataset = fer_custom_dataset(data=self.test_df, 
                                             files_dir=self.frames_dir, 
                                             is_train_dataset=False, 
                                             preload_frames=self.preload_frames, 
                                             balance_dataset=self.balance_dataset,
                                             apply_transformations=self.apply_transformations, 
                                             normalize=self.normalize
                                             )
        print(f"--Dataset-- Test dataset size: {test_dataset.__len__()}")
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
