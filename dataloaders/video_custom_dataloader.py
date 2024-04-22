from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import OVERLAP_SUBJECTS_FRAMES, DF_SPLITTING, RANDOM_SEED
import pandas as pd
from datasets.video_custom_dataset import video_custom_dataset
from shared.constants import mapping_to_positive_negative

class video_custom_dataloader(DataLoader):
    def __init__(self,
                 csv_original_files: str,
                 csv_frames_files: str,
                 batch_size: int,
                 frames_dir: str,
                 seed: int = RANDOM_SEED,
                 limit: int = None,
                 overlap_subjects_frames: bool = True,
                 use_positive_negative_labels: bool = True,
                 preload_frames: bool = True,
                 apply_transformations: bool = True,
                 balance_dataset: bool = True,
                 normalize: bool = True,
                 ):
        self.batch_size = batch_size
        self.data = pd.read_csv(csv_original_files)
        self.csv_frames_files = pd.read_csv(csv_frames_files)
        self.frames_dir = frames_dir
        self.seed = seed
        self.limit = limit
        self.overlap_subjects_frames = overlap_subjects_frames
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

        # Split the dataset into train, validation and test
        if overlap_subjects_frames:
            print("--Dataloader-- Splitting the dataset WITH overlapping between subjects")
            self.split_datasets_overlapping_subjects()
        else:
            print("--Dataloader-- Splitting the dataset WITHOUT overlapping between subjects")
            self.split_datasets_without_overlapping_subjects()

    def split_datasets_overlapping_subjects(self):
        # Split the dataset using the original dataset (video), frame's subjects can overlap between train, val and test
        # Split the subjects using DF_SPLITTING configuration
        self.train_df, temp_df = train_test_split(self.data, test_size=DF_SPLITTING[0], random_state=self.seed)
        self.val_df, self.test_df = train_test_split(temp_df, test_size=DF_SPLITTING[1], random_state=self.seed)

        # Get the subjects for each dataset
        train_subjects = self.train_df["actor"].unique()
        val_subjects = self.val_df["actor"].unique()
        test_subjects = self.test_df["actor"].unique()

        print(f"--- Train subjects: {train_subjects} \n--- Validation subjects: {val_subjects} \n--- Test subjects: {test_subjects}")

        # For each video select its frames from the frames dataset
        # Example:
        # On the original datasetfile_name is: 01-01-01-01-01-01-01.mp4
        # On the frames dataset file_name is: 01-01-01-01-01-01-01_1.png
        # Select all the frames that contain the file_name "01-01-01-01-01-01-01"

        # Create a list of file_name without the extension
        train_file_names = self.train_df["file_name"].apply(lambda x: x.split(".")[0])
        val_file_names = self.val_df["file_name"].apply(lambda x: x.split(".")[0])
        test_file_names = self.test_df["file_name"].apply(lambda x: x.split(".")[0])

        # Load the frames dataset dataset and select the frames that contain the file_name
        self.load_frames_from_file_names(train_file_names, val_file_names, test_file_names)
     
    def split_datasets_without_overlapping_subjects(self):
        # Split the dataset using the original dataset (video), frame's subjects can't overlap between train, val and test
        # x subjects for train, y subjects for val and z subjects for test (where x != y != z)
        # Get the subjects
        subjects = self.data["actor"].unique()
        print(f"--Dataloader-- Subjects: {subjects}")

        # Split the subjects: n - 1 for train, 1 for val and 1 for test
        print("--Dataloader-- Splitting the dataset using n-1 subjects for train, 1 for val and 1 for test")
        train_subjects, tmp_subjects = train_test_split(subjects, test_size=0.05, random_state=self.seed)
        val_subjects, test_subjects = train_test_split(tmp_subjects, test_size=0.5, random_state=self.seed)

        print(f"Train subjects: {train_subjects} \nValidation subjects: {val_subjects} \nTest subjects: {test_subjects}")

        # For each actor select its frames from the frames dataset
        # Example:
        # On the original datasetfile_name is: 01-01-01-01-01-01-01.mp4
        # On the frames dataset file_name is: 01-01-01-01-01-01-01_1.png
        # Select all the frames that contain "*-01_*" in the file_name (actor 01)

        # Create a list of file_name without the extension
        train_file_names = self.data[self.data["actor"].isin(train_subjects)]["file_name"].apply(lambda x: x.split(".")[0])
        val_file_names = self.data[self.data["actor"].isin(val_subjects)]["file_name"].apply(lambda x: x.split(".")[0])
        test_file_names = self.data[self.data["actor"].isin(test_subjects)]["file_name"].apply(lambda x: x.split(".")[0])

        # Load the frames dataset dataset and select the frames that contain the file_name
        self.load_frames_from_file_names(train_file_names, val_file_names, test_file_names)
    
    def load_frames_from_file_names(self, train_file_names, val_file_names, test_file_names):
        # Load the frames dataset dataset and select the frames that contain the file_name
        self.train_df = self.csv_frames_files[self.csv_frames_files["file_name"].apply(lambda x: x.split("_")[0]).isin(train_file_names)]
        self.val_df = self.csv_frames_files[self.csv_frames_files["file_name"].apply(lambda x: x.split("_")[0]).isin(val_file_names)]
        self.test_df = self.csv_frames_files[self.csv_frames_files["file_name"].apply(lambda x: x.split("_")[0]).isin(test_file_names)]

        # Map the emotions to positive/negative labels
        if self.use_positive_negative_labels:
            print("--Dataloader-- Using positive/negative labels mapping.")
            self.train_df.loc[:, 'emotion'] = self.train_df['emotion'].astype(int).map(mapping_to_positive_negative)
            self.val_df.loc[:, 'emotion'] = self.val_df['emotion'].astype(int).map(mapping_to_positive_negative)
            self.test_df.loc[:, 'emotion'] = self.test_df['emotion'].astype(int).map(mapping_to_positive_negative)

        # Print unique values in the column 'emotion'
        print(f"--- Train emotions: {self.train_df['emotion'].unique()} \n--- Validation emotions: {self.val_df['emotion'].unique()} \n--- Test emotions: {self.test_df['emotion'].unique()}")

    def get_train_dataloader(self):
        train_dataset = video_custom_dataset(data=self.train_df, 
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
        print(f"--Dataset-- Test dataset size: {test_dataset.__len__()}")
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
