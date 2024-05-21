from config import (
    DATA_DIR,
    LENGTH,
    RANDOM_SEED,
    WT,
    BALANCE_DATASET
)
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List
import os
import torch
import numpy as np
import pandas as pd
from datasets.DEAP_dataset import DEAPDataset
from utils.ppg_utils import fft, detrend, bandpass_filter, moving_average_filter
from sklearn.model_selection import train_test_split
from packages.rppg_toolbox.utils.plot import plot_signal
import warnings
warnings.filterwarnings('ignore')


PLOT_DEBUG_INDEX = 1
BAD_SIGNAL_INDEX = 10

class DEAPDataLoader(DataLoader):
    def __init__(self,
                 batch_size: int):
        self.batch_size = batch_size
        self.data = self.load_data()
        self.data["valence"] = self.data["valence"].apply(lambda x: self.discretize_labels(torch.tensor(x)))
        # plot_signal(self.data["ppg"].iloc[PLOT_DEBUG_INDEX], "Original Signal")
        print("Detrending signal...")
        self.data["ppg"] = self.data["ppg"].apply(lambda x: detrend(np.array(x)))
        # plot_signal(self.data["ppg"].iloc[PLOT_DEBUG_INDEX], "Detrended Signal")
        print("Bandpass filtering...")
        self.data["ppg"] = self.data["ppg"].apply(lambda x: bandpass_filter(np.array(x)))
        # plot_signal(self.data["ppg"].iloc[PLOT_DEBUG_INDEX], "Bandpass Filtered Signal")
        print("Moving average filtering...")
        self.data["ppg"] = self.data["ppg"].apply(lambda x: moving_average_filter(x))
        # plot_signal(self.data["ppg"].iloc[PLOT_DEBUG_INDEX], "Moving Average Filtered Signal")

         
        # NOISE_THRESHOLD = 0.002
        # print("plotting good and bad signals...")
        # bad_signal_plot = 0
        # good_signal_plot = 0
        # for i, row in self.data.iterrows():
        #     if bad_signal_plot > 5 and good_signal_plot > 5:
        #         break
        #     ppg, stn = row["ppg"], row["stn"]
        #     print(f"stn for i: {i} is: {stn}")
        #     if stn >= NOISE_THRESHOLD and good_signal_plot <=5:
        #         plot_signal(np.array(ppg), f"debug_plots/good_signal_{good_signal_plot}_{i}")
        #         good_signal_plot +=1 
        #     if stn < NOISE_THRESHOLD and bad_signal_plot <= 5:
        #         plot_signal(np.array(ppg), f"debug_plots/bad_signal_{bad_signal_plot}_{i}")
        #         bad_signal_plot += 1
        
        # print(f"Data before filtering bad signals: {len(self.data)}")
        # self.data = self.data.query(f'stn > {NOISE_THRESHOLD}')
        # print(f"Data after filtering bad signals: {len(self.data)}")


        print("Performing min-max normalization")
        self.data = self.normalize_data(self.data)

        ppgs = np.stack(self.data["ppg"].to_numpy())
        mean, std = ppgs.mean(), ppgs.std()
        print(f"DEAP mean and std are: {mean, std}")

        print("Standardizing data") 
        ppgs = np.stack(self.data["ppg"].to_numpy())
        mean, std = ppgs.mean(), ppgs.std()
        print(f"DEAP mean and std are: {mean, std}")
        self.data["ppg"] = self.data["ppg"].apply(lambda x: (x-mean)/std) 
        ppgs = np.stack(self.data["ppg"].to_numpy())
        mean, std = ppgs.mean(), ppgs.std()
        print(f"DEAP mean and std are: {mean, std}")



        print("Splitting the data")
        self.train_df, self.val_df, self.test_df = self.split_data()

        print("Slicing the data")
        self.train_df = self.slice_data(self.train_df)
        self.val_df = self.slice_data(self.val_df)
        self.test_df = self.slice_data(self.test_df)
        


        if WT:
            print(f"Performing FFT...")
            tqdm.pandas()
            self.train_df["ppg"] = self.train_df["ppg"].progress_apply(fft)
            self.val_df["ppg"] = self.val_df["ppg"].progress_apply(fft)
            self.test_df["ppg"] = self.test_df["ppg"].progress_apply(fft)
        else:
            print("Skipped FFT")
        
        if BALANCE_DATASET:
            self.balance_data()

        print(f"Train_df length: {len(self.train_df)}")
        print(f"Val_df length: {len(self.val_df)}")
        print(f"Test_df length: {len(self.test_df)}")
    


    def balance_data(self):
        label_counts = self.train_df["valence"].value_counts()
        print("Count before (train)", label_counts)
        target_count = label_counts.min()

        def balanced_sample(group):
            return group.sample(target_count, random_state=RANDOM_SEED)

        self.train_df = self.train_df.groupby('valence').apply(balanced_sample)
        print("Count after (train)", self.train_df["valence"].value_counts())

        # label_counts = self.val_df["valence"].value_counts()
        # target_count = label_counts.min()
        # print("Count before (val)", label_counts)
        # self.val_df = self.val_df.groupby('valence').apply(balanced_sample)
        # print("Count after (val)", self.val_df["valence"].value_counts())

        # label_counts = self.test_df["valence"].value_counts()
        # target_count = label_counts.min()
        # print("Count before (val)", label_counts)
        # self.test_df = self.test_df.groupby('valence').apply(balanced_sample)
        # print("Count after (test)", self.test_df["valence"].value_counts())

        

    def load_data(self) -> pd.DataFrame:
        data_dir = os.path.join(DATA_DIR, "DEAP", "data")
        ppg_channel = 38
        df = []
        for i, file in enumerate(os.listdir(data_dir)):
            if not file.endswith(".dat"):
                continue
            abs_path = os.path.join(data_dir, file)
            print(f"reading file: {abs_path}: subject is {i}")
            with open(abs_path, "rb") as f:
                # resolve the python 2 data problem by encoding : latin1
                subject = pickle.load(f, encoding='latin1')
                for trial_i in range(40):
                    valence: np.ndarray = subject["labels"][trial_i][0] #index 0 is valence, 1 arousal
                    data: np.ndarray = subject["data"][trial_i][ppg_channel]
                    df.append({"ppg": data, "valence": valence, "subject": i})
        return pd.DataFrame(df)
    

    def normalize_data(self, df):
        """
        Compute the min and max value per subject and uses min-max scaling in order to normalize the data.
        """
        min_max_mapping = {}
        for i, row in df.iterrows():
            ppg, subject = row["ppg"], row["subject"]
            _min, _max = min(ppg), max(ppg)
            if subject not in min_max_mapping:
                min_max_mapping[subject] = {"_min": _min, "_max": _max}
            else:
                if _min < min_max_mapping[subject]["_min"]:
                    min_max_mapping[subject]["_min"] =  _min
                if _max > min_max_mapping[subject]["_max"]:
                    min_max_mapping[subject]["_max"] = _max
        
        a, b = [0,1]
        new_df = []
        for i, row in df.iterrows():
            ppg, subject, valence = row["ppg"], row["subject"], row["valence"]
            # _min, _max = min_max_mapping[subject]["_min"], min_max_mapping[subject]["_max"] 
            _min, _max = min(ppg), max(ppg)
            ppg = a + ((ppg - _min)*(b-a)) / (_max - _min)
            new_df.append({"ppg": ppg, "valence": valence, "subject": subject})
        return pd.DataFrame(new_df)


    def get_mean_std(self, data):
        cat_data = data["ppg"].to_numpy()
        mean, std = cat_data.mean(), cat_data.std()
        return mean, std


    def standardize_data(self, data, mean=None, std=None):
        if mean is not None and std is not None:
            data["ppg"] = data["ppg"].apply(lambda x: (x-mean)/std) 
            mean, std = self.get_mean_std(data)
            print(f"Normalized DEAP mean + std: {mean, std}")
            return data
        #Standardize
        og_mean, og_std = self.get_mean_std(data)
        print(f"Before DEAP mean + std: {og_mean, og_std}")
        data["ppg"] = data["ppg"].apply(lambda x: (x-og_mean)/og_std) 
        return data, (og_mean, og_std)
       
    def split_data(self):
        train_df, temp_df = train_test_split(self.data, test_size=0.2, random_state=RANDOM_SEED, stratify=self.data["valence"])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["valence"])
        return train_df, val_df, test_df

    def slice_ppg_window(self, ppg_signal,peak_index, window_size):
      """
      Slices a window from the PPG signal to contain a single pulse with the peak at the center.
      """
      window_size = min(window_size, len(ppg_signal))
      half_window_size = window_size // 2
      # Check if the peak is close enough to the edges to fit the entire window
      if peak_index < half_window_size or peak_index >= len(ppg_signal) - half_window_size:
        # If not, center the window as much as possible
        start_index = max(0, peak_index - half_window_size + 1)
        end_index = min(len(ppg_signal), peak_index + half_window_size)
      else:
        # Center the window perfectly
        start_index = peak_index - half_window_size
        end_index = start_index + window_size
      # Slice the window from the signal
      sliced_window = ppg_signal[start_index:end_index]
      return sliced_window 

    def slice_ppg_windows(self, ppg_signal, window_size):
          """
          Slices the entire PPG signal into windows containing single pulses.
          """

          # Find all potential peak indices
          potential_peaks = np.diff(np.sign(np.diff(ppg_signal))) > 0  # Identify rising edges

          windows = []
          PEAK_STEP = 1
          for i in range(0, len(potential_peaks.nonzero()[0]), PEAK_STEP):
            peak_index = potential_peaks.nonzero()[0][i]
            sliced_window = self.slice_ppg_window(ppg_signal, peak_index, window_size)
            windows.append(sliced_window)

          return windows    

    def slice_data(self, df, length=LENGTH) -> pd.DataFrame:
        if length > 8064:
            raise ValueError(f"Length cannot be greater than original length")
        new_df = []
        for row_i, row in df.iterrows():
            ppg, label, subject = row["ppg"], row["valence"], row["subject"]
            ppg_segments = self.slice_ppg_windows(ppg, window_size=length)
            for i, ppg_segment in enumerate(ppg_segments):
                if len(ppg_segment) != length:
                    continue

                #Remove low quality sliced that comes from low quality signal
                if len((np.diff(np.sign(np.diff(ppg_segment))) > 0).nonzero()[0]) > 4:
                    # plot_signal(ppg_segment, f"debug_plots/bad_slices/slice_{i}_{row_i}")
                    # plot_signal(ppg, f"debug_plots/bad_slices/whole_slice_{row_i}")
                    continue 
                
                # Only keep signals that have the peak on the center
                if not 30 <= np.argmax(ppg_segment) <= 80:
                    continue
        
                # plot_signal(ppg_segment, f"debug_plots/slices/after_slice_{i}_{row_i}")
                # plot_signal(ppg, f"debug_plots/slices/whole_slice_{row_i}")
                
                new_row = {
                        "ppg": ppg_segment,
                        "valence": label,
                        "subject": subject}
                new_df.append(new_row)
        new_df = pd.DataFrame(new_df)
        return new_df
    
    def discretize_labels(self, valence: torch.Tensor) -> List[float]:
        self.labels = torch.full_like(valence, -1)
        self.labels[(valence >= 1) & (valence < 2) ] = 0 
        self.labels[(valence >= 2) & (valence < 7) ] = 1
        self.labels[(valence >= 7) & (valence <= 9) ] = 2
        # self.labels[(valence >= 1) & (valence < 3) ] = 0 
        # self.labels[(valence >= 3) & (valence < 6) ] = 1
        # self.labels[(valence >= 6) & (valence <= 9) ] = 2
        return self.labels.tolist()

    def get_train_dataloader(self):
        dataset = DEAPDataset(self.train_df)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def get_val_dataloader(self):
        dataset = DEAPDataset(self.val_df)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def get_test_dataloader(self):
        dataset = DEAPDataset(self.test_df)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    # Test the dataloader
    dataloader = DEAPDataLoader(batch_size=32)
