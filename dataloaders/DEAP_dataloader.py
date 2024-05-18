from config import (
    DATA_DIR,
    LENGTH,
    RANDOM_SEED,
    STEP,
    WT
)
import pickle
from shared.constants import CEAP_MEAN, CEAP_STD
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List
import os
import torch
import numpy as np
import pandas as pd
from datasets.DEAP_dataset import DEAPDataset
from utils.ppg_utils import wavelet_transform, stft, signaltonoise, second_derivative
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from scipy import signal
from packages.rppg_toolbox.utils.plot import plot_signal
import warnings
from scipy.signal import butter, filtfilt
warnings.filterwarnings('ignore')

# Sample rate is 128hz
# Signals are 8064 long
# Each signal is 63 seconds long


PLOT_DEBUG_INDEX = 1
BAD_SIGNAL_INDEX = 10

class DEAPDataLoader(DataLoader):
    def __init__(self,
                 batch_size: int):
        self.batch_size = batch_size
        self.data = self.load_data()
        self.data["valence"] = self.data["valence"].apply(lambda x: self.discretize_labels(torch.tensor(x)))
        plot_signal(self.data["ppg"].iloc[PLOT_DEBUG_INDEX], "original")
        print("Detrending signal...")
        self.data["ppg"] = self.data["ppg"].apply(lambda x: self.detrend_ppg(np.array(x)))
        plot_signal(self.data["ppg"].iloc[PLOT_DEBUG_INDEX], "detrended")
        print("Bandpass filtering...")
        self.data["ppg"] = self.data["ppg"].apply(lambda x: self.bandpass_filter_ppg(np.array(x)))
        plot_signal(self.data["ppg"].iloc[PLOT_DEBUG_INDEX], "bandpass_filtered")
        print("Moving average filtering...")
        self.data["ppg"] = self.data["ppg"].apply(lambda x: self.moving_average_filter(x))
        plot_signal(self.data["ppg"].iloc[PLOT_DEBUG_INDEX], "moving_average_filtered")
        
        self.data["stn"] = self.data["ppg"].apply(lambda x: signaltonoise(x))
        
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
        
        self.train_df, self.val_df, self.test_df = self.split_data()
        self.train_df = self.slice_data(self.train_df)
        self.val_df = self.slice_data(self.val_df)
        self.test_df = self.slice_data(self.test_df)

        if WT:
            print(f"Performing wavelet transform...")
            tqdm.pandas()
            self.train_df["ppg"] = self.train_df["ppg"].progress_apply(wavelet_transform)
            self.val_df["ppg"] = self.val_df["ppg"].progress_apply(wavelet_transform)
            self.test_df["ppg"] = self.test_df["ppg"].progress_apply(wavelet_transform)
        else:
            print("Skipped wavelet transform")

        print(f"Train_df length: {len(self.train_df)}")
        print(f"Val_df length: {len(self.val_df)}")
        print(f"Test_df length: {len(self.test_df)}")

        # target_count = label_counts.min()
        # # Sample function to get balanced sample from each group
        # def balanced_sample(group):
        #   return group.sample(target_count, random_state=RANDOM_SEED)
        # # Apply sample function to each group in the DataFrame 
        # self.data = self.data.groupby('valence').apply(balanced_sample)

    def load_data(self) -> pd.DataFrame:
        data_dir = os.path.join(DATA_DIR, "DEAP", "data")
        # metadata_dir = os.path.join(DATA_DIR, "DEAP", "metadata")
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
                    # NOTE: valence is in range [1,9]
                    valence: np.ndarray = subject["labels"][trial_i][0] #index 0 is valence, 1 arousal
                    data: np.ndarray = subject["data"][trial_i][ppg_channel]
                    df.append({"ppg": data, "valence": valence, "subject": i})
        return pd.DataFrame(df)
    
    def moving_average_filter(self, data, window_size=10):
      """
      Applies a moving average filter to a 1D NumPy array.

      Args:
          data: The 1D NumPy array to filter.
          window_size: The size of the moving average window (positive integer).

      Returns:
          The filtered 1D NumPy array.
      """

      # Check for valid window size
      if window_size <= 0:
        raise ValueError("Window size must be a positive integer.")

      # Calculate the number of elements to pad at the beginning and end
      pad_size = window_size // 2

      # Pad the data with mirrored values at the beginning and end
      padded_data = np.concatenate((data[:pad_size][::-1], data, data[-pad_size:][::-1]))

      # Apply moving average using convolution
      smoothed_data = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')

      return smoothed_data

    def bandpass_filter_ppg(self, data, fs=128, lowcut=0.5, highcut=20, order=5):
      """
      Bandpass filters a PPG signal using a Butterworth filter.

      Args:
          data: The PPG signal as a numpy array.
          fs: The sampling frequency of the signal in Hz.
          lowcut: Lower cutoff frequency of the bandpass filter in Hz.
          highcut: Higher cutoff frequency of the bandpass filter in Hz.
          order: The order of the Butterworth filter (default: 5).

      Returns:
          The filtered PPG signal as a numpy array.
      """

      nyquist = 0.5 * fs
      lowcut_norm = lowcut / nyquist
      highcut_norm = highcut / nyquist

      # Design the Butterworth filter
      b, a = butter(order, [lowcut_norm, highcut_norm], btype='band')

      # Apply the filter twice (filtfilt) for zero-phase filtering
      filtered_data = filtfilt(b, a, data)

      return filtered_data

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
        
        # alpha = 1000
        alpha = 1
        a, b = [-1,1]
        new_df = []
        for i, row in df.iterrows():
            ppg, subject, valence = row["ppg"], row["subject"], row["valence"]
            # _min, _max = min_max_mapping[subject]["_min"], min_max_mapping[subject]["_max"] 
            _min, _max = min(ppg), max(ppg)
            ppg = (a + ((ppg - _min)*(b-a)) / (_max - _min)) * alpha
            new_df.append({"ppg": ppg, "valence": valence, "subject": subject})
        return pd.DataFrame(new_df)


    def get_mean_std(self):
        cat_data = np.concatenate(self.data["ppg"], axis=0)
        mean, std = cat_data.mean(), cat_data.std()
        return mean, std

    def standardize_data(self):
        #Standardize
        self.data["ppg"] = self.data["ppg"].apply(lambda x: np.array(x))
        print(f"Shape of signal is: {self.data['ppg'].iloc[0].shape}")
        mean, std = self.get_mean_std()
        print(f"Before DEAP mean + std: {mean, std}")
        self.data["ppg"] = self.data["ppg"].apply(lambda x: (x-mean)/std) 
        mean, std = self.get_mean_std()
        print(f"Normalized DEAP mean + std: {mean, std}")

    def detrend_ppg(self, ppg_signal):
        x = np.linspace(0, ppg_signal.shape[0], ppg_signal.shape[0])
        # print(f"x shape is: {x.shape}, while ppg_signal shape is {ppg_signal.shape}")
        model = np.polyfit(x, ppg_signal, 50)
        predicted = np.polyval(model, x)
        return ppg_signal - predicted
        return signal.detrend(ppg_signal)

       
    def split_data(self):
        train_df, temp_df = train_test_split(self.data, test_size=0.2, random_state=RANDOM_SEED)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED)
        return train_df, val_df, test_df

    def slice_ppg_window(self, ppg_signal,peak_index, window_size):
      """
      Slices a window from the PPG signal to contain a single pulse with the peak at the center.

      Args:
        ppg_signal: A NumPy array representing the filtered PPG signal.
        window_size: The desired size of the window (number of data points).

      Returns:
        A NumPy array containing the sliced window with the peak at the center.
      """
      # Ensure the window size is less than or equal to the signal length
      window_size = min(window_size, len(ppg_signal))
      # Calculate the half window size (assuming an even window size for peak centering)
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

    def slice_ppg_windows(self, ppg_signal, window_size, overlap=0):
          """
          Slices the entire PPG signal into windows containing single pulses.

          Args:
            ppg_signal: A NumPy array representing the filtered PPG signal.
            window_size: The desired size of the window (number of data points).
            overlap (optional): The number of data points by which consecutive windows overlap.

          Returns:
            A list of NumPy arrays, where each array represents a window with a single pulse.
          """

          # Find all potential peak indices
          potential_peaks = np.diff(np.sign(np.diff(ppg_signal))) > 0  # Identify rising edges

          # Create an empty list to store windows
          windows = []

          # Iterate through potential peaks
          for peak_index in potential_peaks.nonzero()[0]:
            # Call the slice function for each peak
            sliced_window = self.slice_ppg_window(ppg_signal, peak_index, window_size)
            windows.append(sliced_window)

          # Handle overlapping windows (optional)
          if overlap > 0:
            # Adjust window start indices for overlapping windows
            for i in range(1, len(windows)):
              windows[i][0] = max(windows[i][0], windows[i-1][window_size - overlap])

          return windows    

    #TODO: take a single pulse for each window, and align it in the center
    def slice_data(self, df, length=LENGTH, step=STEP) -> pd.DataFrame:
        if length > 3000:
            raise ValueError(f"Length cannot be greater than original length")
        new_df = []
        for row_i, row in df.iterrows():
            ppg, label, subject = row["ppg"], row["valence"], row["subject"]
            if row_i == PLOT_DEBUG_INDEX:
                plot_signal(ppg, "before_slice")
            ppg_segments = self.slice_ppg_windows(ppg, window_size=length)
            for i, ppg_segment in enumerate(ppg_segments):
                if len(ppg_segment) != length:
                    continue

                #Remove low quality sliced that comes from low quality signal
                if len((np.diff(np.sign(np.diff(ppg_segment))) > 0).nonzero()[0]) > 3:
                    # plot_signal(ppg_segment, f"debug_plots/bad_slices/slice_{i}")
                    continue 
                
                # Only keep signals that have the peak on the center
                if not 30 <= np.argmax(ppg_segment) <= 80:
                    continue

        
                # plot_signal(ppg_segment, f"debug_plots/slices/after_slice_{i}_{row_i}")

                new_row = {
                        "ppg": ppg_segment ,
                        "valence": label,
                        "subject": subject}
                new_df.append(new_row)
        new_df = pd.DataFrame(new_df)
        return new_df
    
    def discretize_labels(self, valence: torch.Tensor) -> List[float]:
        self.labels = torch.full_like(valence, -1)
        self.labels[(valence >= 1) & (valence <= 3) ] = 0 
        self.labels[(valence >= 3) & (valence <= 6) ] = 1
        self.labels[(valence >= 6) & (valence <= 9) ] = 2
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
    # test_loader = dataloader.get_test_dataloader()

    # for i, data in enumerate(test_loader):
    #     print(f"Data size is {len(data)}")
    #     break
