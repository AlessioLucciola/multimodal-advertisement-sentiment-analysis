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

        # Test 1: Split the dataset using the original dataset (video)
        # Split the subjects using DF_SPLITTING configuration
        self.train_df, temp_df = train_test_split(self.data, test_size=DF_SPLITTING[0], random_state=self.seed)
        self.val_df, self.test_df = train_test_split(temp_df, test_size=DF_SPLITTING[1], random_state=self.seed)

        # Get the subjects for each dataset
        train_subjects = self.train_df["actor"].unique()
        val_subjects = self.val_df["actor"].unique()
        test_subjects = self.test_df["actor"].unique()

        print(f"Train subjects: {train_subjects} \nValidation subjects: {val_subjects} \nTest subjects: {test_subjects}")

        # For each video select its frames from the frames dataset
        # Example:
        # On the original datasetfile_name is: 01-01-01-01-01-01-01.mp4
        # On the frames dataset file_name is: 01-01-01-01-01-01-01_1.png
        # Select all the frames that contain the file_name "01-01-01-01-01-01-01"

        # create a list of file_name without the extension
        train_file_names = self.train_df["file_name"].apply(lambda x: x.split(".")[0])
        val_file_names = self.val_df["file_name"].apply(lambda x: x.split(".")[0])
        test_file_names = self.test_df["file_name"].apply(lambda x: x.split(".")[0])

        # Load the frames dataset dataset and select the frames that contain the file_name
        frames_data = pd.read_csv(VIDEO_METADATA_FRAMES_CSV)
        self.train_df = frames_data[frames_data["file_name"].apply(lambda x: x.split("_")[0]).isin(train_file_names)]
        self.val_df = frames_data[frames_data["file_name"].apply(lambda x: x.split("_")[0]).isin(val_file_names)]
        self.test_df = frames_data[frames_data["file_name"].apply(lambda x: x.split("_")[0]).isin(test_file_names)]

        print(f"--Dataloader-- Train dataset size: {self.train_df.__len__()} | Validation dataset size: {self.val_df.__len__()} | Test dataset size: {self.test_df.__len__()}")
        
        # -----------------------------------------
        
        # Test 2: Split the dataset using the original dataset (video), x subjects for train, y subjects for val and z subjects for test (where x != y != z)
        # # Get the subjects
        # subjects = self.data["actor"].unique()
        # print(f"--Dataloader-- Subjects: {subjects}")

        # # Split the subjects using DF_SPLITTING configuration
        # train_subjects, temp_subjects = train_test_split(subjects, test_size=DF_SPLITTING[0], random_state=self.seed)
        # val_subjects, test_subjects = train_test_split(temp_subjects, test_size=DF_SPLITTING[1], random_state=self.seed)

        # # Split the subjects: n - 1 for train, 1 for val and 1 for test
        # # train_subjects, tmp_subjects = train_test_split(subjects, test_size=0.05, random_state=self.seed)
        # # val_subjects, test_subjects = train_test_split(tmp_subjects, test_size=0.5, random_state=self.seed)

        # print(f"Train subjects: {train_subjects} \nValidation subjects: {val_subjects} \nTest subjects: {test_subjects}")

        # # For each actor select its frames from the frames dataset
        # # Example:
        # # On the original datasetfile_name is: 01-01-01-01-01-01-01.mp4
        # # On the frames dataset file_name is: 01-01-01-01-01-01-01_1.png
        # # Select all the frames that contain "*-01_*" in the file_name (actor 01)

        # # create a list of file_name without the extension
        # train_file_names = self.data[self.data["actor"].isin(train_subjects)]["file_name"].apply(lambda x: x.split(".")[0])
        # val_file_names = self.data[self.data["actor"].isin(val_subjects)]["file_name"].apply(lambda x: x.split(".")[0])
        # test_file_names = self.data[self.data["actor"].isin(test_subjects)]["file_name"].apply(lambda x: x.split(".")[0])

        # # Save to .csv the selected file names
        # train_file_names.to_csv("train_file_names.csv", index=False)
        # val_file_names.to_csv("val_file_names.csv", index=False)
        # test_file_names.to_csv("test_file_names.csv", index=False)

        # # Load the frames dataset dataset and select the frames that contain the file_name
        # frames_data = pd.read_csv(VIDEO_METADATA_FRAMES_CSV)
        # self.train_df = frames_data[frames_data["file_name"].apply(lambda x: x.split("_")[0]).isin(train_file_names)]
        # self.val_df = frames_data[frames_data["file_name"].apply(lambda x: x.split("_")[0]).isin(val_file_names)]
        # self.test_df = frames_data[frames_data["file_name"].apply(lambda x: x.split("_")[0]).isin(test_file_names)]

        # print(f"--Dataloader-- Train dataset size: {self.train_df.__len__()} | Validation dataset size: {self.val_df.__len__()} | Test dataset size: {self.test_df.__len__()}")

    def get_train_dataloader(self):
        train_dataset = video_custom_dataset(data=self.train_df, 
                                              files_dir=self.frames_dir, 
                                              is_train_dataset=True, 
                                              preload_frames=self.preload_frames, 
                                              balance_dataset=self.balance_dataset, 
                                              apply_transformations=self.apply_transformations, 
                                              normalize=self.normalize
                                              )
        print(f"--Dataset-- Validation dataset size: {train_dataset.__len__()}")
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
        print(f"--Dataset-- Validation dataset size: {val_dataset.__len__()}")
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
        print(f"--Dataset-- Validation dataset size: {test_dataset.__len__()}")
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
