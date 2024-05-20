import pywt
import torch
import numpy as np
import pandas as pd
from biosppy.signals import bvp
from config import WAVELET_STEP, LENGTH
from scipy.signal import welch, filtfilt, butter

def statistical_features(x: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(x, dim=1).unsqueeze(0)
    std = torch.std(x, dim=1).unsqueeze(0)
    skewness = torch.tensor([pd.Series(x[0]).skew()]).unsqueeze(0)
    kurtosis = torch.tensor([pd.Series(x[0]).kurtosis()]).unsqueeze(0)
    onsets, hr = onsets_and_hr(x)
    onsets = onsets.mean()
    hr = hr.mean()
    return torch.cat((mean, std, skewness, kurtosis, onsets, hr), dim=0)

def wavelet_transform(x: torch.Tensor | np.ndarray):
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    # return np.zeros((LENGTH // WAVELET_STEP,  LENGTH))
    scales = np.arange(1, len(x) + 1, WAVELET_STEP)
    coef, _ = pywt.cwt(x, scales, "mexh", method="conv")
    # print(f"coef shape is {coef.shape}")
    return coef

def stft(x: np.ndarray):
    if isinstance(x, list):
        x = np.array(x)

    x_torch = torch.from_numpy(x)
    win_length = 64
    window = torch.signal.windows.hamming(win_length)
    res = torch.stft(input=x_torch, 
                     n_fft=64,
                     hop_length=2, 
                     return_complex=True,
                     window=window, 
                     win_length=win_length)
    magnitude = torch.abs(res)
    phase = torch.angle(res)
    return torch.stack((magnitude, phase), dim=0)

def fft(x: np.ndarray):
    if isinstance(x, list):
        x = np.array(x)

    x_torch = torch.from_numpy(x)
    res = torch.fft.fft(x_torch)
    magnitude = torch.abs(res)
    phase = torch.angle(res)
    return torch.stack((res.real, magnitude, phase), dim=0)

def onsets_and_hr(x):
    features = bvp.bvp(x, 100, show=False)
    onsets = torch.tensor(features["onsets"]).unsqueeze(0)
    hr = torch.tensor(features["heart_rate"]).unsqueeze(0)
    delta = onsets.shape[1] - hr.shape[1]
    onsets = onsets[:, :-delta]
    return torch.cat((onsets, hr), dim=0)

def power_spectrum(ppg_data, fs=128):
  """
  Calculates the power spectrum of a PPG signal.

  Args:
      ppg_data: The PPG signal as a NumPy array.
      fs: The sampling frequency of the signal in Hz.

  Returns:
      A tuple containing:
          - frequencies: An array of frequencies corresponding to the power spectrum.
          - power_density: An array of power spectral density values.
  """

  # Welch's method is a common approach for estimating power spectra
  frequencies, power_density = welch(ppg_data, fs=fs, nperseg=1024, window='hann')

  return frequencies, power_density


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def second_derivative(signal):
    x = np.linspace(0, signal.shape[0], num=signal.shape[0])
    dfdx = np.gradient(signal, x)
    d2fdx2 = np.gradient(dfdx, x)
    return d2fdx2


if __name__ == "__main__":
    pass
    # x = np.random.rand(1000)
    # features = differential_entropy(x)
    # print(f"features: {features} with shape {features.shape}")
