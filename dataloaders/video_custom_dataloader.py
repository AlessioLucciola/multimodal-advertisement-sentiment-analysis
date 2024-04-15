from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import DF_SPLITTING, RANDOM_SEED, VIDEO_METADATA_FRAMES_CSV
import pandas as pd
from datasets.video_custom_dataset import video_custom_dataset

class video_custom_dataloader(DataLoader):
    def __init__(self,
                 csv_file: str,
                 batch_size: int,
                 frames_dir: str,
                 seed: int = RANDOM_SEED,
                 limit: int = None,
                 preload_frames: bool = True,
                 apply_transformations: bool = True,
                 balance_dataset: bool = True,
                 normalize: bool = True,
                 ):
        self.batch_size = batch_size
        self.data = pd.read_csv(csv_file)
        self.frames_dir = frames_dir
        self.seed = seed
        self.limit = limit
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

        # Split the dataset using the original dataset (video)
        self.train_df, temp_df = train_test_split(self.data, test_size=DF_SPLITTING[0], random_state=self.seed)
        self.val_df, self.test_df = train_test_split(temp_df, test_size=DF_SPLITTING[1], random_state=self.seed)

        # For each video select its frames from the frames dataset
        # Example:
        # On the original datasetfile_name is: 01-01-01-01-01-01-01.mp4
        # On the frames dataset file_name is: 01-01-01-01-01-01-01_1.png
        # Select all the frames that contain the file_name "01-01-01-01-01-01-01"

        # create a list of file_name without the extension
        train_file_names = self.data["file_name"].apply(lambda x: x.split(".")[0])
        val_file_names = self.val_df["file_name"].apply(lambda x: x.split(".")[0])
        test_file_names = self.test_df["file_name"].apply(lambda x: x.split(".")[0])

        # Load the frames dataset dataset and select the frames that contain the file_name
        frames_data = pd.read_csv(VIDEO_METADATA_FRAMES_CSV)
        self.train_df = frames_data[frames_data["file_name"].apply(lambda x: x.split("_")[0]).isin(train_file_names)]
        self.val_df = frames_data[frames_data["file_name"].apply(lambda x: x.split("_")[0]).isin(val_file_names)]
        self.test_df = frames_data[frames_data["file_name"].apply(lambda x: x.split("_")[0]).isin(test_file_names)]

    def get_train_dataloader(self):
        train_dataset = video_custom_dataset(data=self.train_df, 
                                              files_dir=self.frames_dir, 
                                              is_train_dataset=True, 
                                              preload_frames=self.preload_frames, 
                                              balance_dataset=self.balance_dataset, 
                                              apply_transformations=self.apply_transformations, 
                                              normalize=self.normalize
                                              )
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_val_dataloader(self):
        val_dataset = video_custom_dataset(data=self.val_df, 
                                            files_dir=self.frames_dir, 
                                            is_train_dataset=False, 
                                            preload_frames=self.preload_frames, 
                                            balance_dataset=self.balance_dataset, 
                                            apply_transformations=self.apply_transformations, 
                                            normalize=self.normalize
                                            )
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def get_test_dataloader(self):
        test_dataset = video_custom_dataset(data=self.test_df, 
                                             files_dir=self.frames_dir, 
                                             is_train_dataset=False, 
                                             preload_frames=self.preload_frames, 
                                             balance_dataset=self.balance_dataset,
                                             apply_transformations=self.apply_transformations, 
                                             normalize=self.normalize
                                             )
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
