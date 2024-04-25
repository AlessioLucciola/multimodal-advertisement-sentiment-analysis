import pywt
import torch
import numpy as np
from pyteap.signals.bvp import get_bvp_features
import pandas as pd
from scipy.signal import resample
from scipy.signal.windows import gaussian
from scipy.signal import ShortTimeFFT
from biosppy.signals import bvp
from config import WAVELET_STEP, LENGTH



def statistical_features(x: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(x, dim=1).unsqueeze(0)
    std = torch.std(x, dim=1).unsqueeze(0)
    skewness = torch.tensor([pd.Series(x[0]).skew()]).unsqueeze(0)
    kurtosis = torch.tensor([pd.Series(x[0]).kurtosis()]).unsqueeze(0)
    onsets, hr = onsets_and_hr(x)
    onsets = onsets.mean()
    hr = hr.mean()
    return torch.cat((mean, std, skewness, kurtosis, onsets, hr), dim=0)


def wavelet_transform(x):
    # return np.zeros((2000 // scale_step,  2000))  # TODO: for debug
    # sr = 100 # sample rate in Hz
    scales = np.arange(1, len(x) + 1, WAVELET_STEP)
    coef, freqs = pywt.cwt(x, scales, "morl")
    # print(f"coef shape is {coef.shape}")
    return coef


def stft(x: np.ndarray):
    x_torch = torch.from_numpy(x)
    win_length = 128
    window = torch.signal.windows.hamming(win_length)
    res = torch.stft(input=x_torch, 
                     n_fft=255,
                     hop_length=64, 
                     return_complex=True,
                     window=window, 
                     win_length=win_length)
    real = res.real
    magnitude = torch.abs(res)
    phase = torch.atan2(res.imag, res.real)
    out = torch.stack((real, magnitude, phase), dim=0)
    # print(f"out shape: {out.shape}")
    # print(f"real part shape is {real.shape}")
    return out

def onsets_and_hr(x):
    features = bvp.bvp(x, 100, show=False)
    onsets = torch.tensor(features["onsets"]).unsqueeze(0)
    hr = torch.tensor(features["heart_rate"]).unsqueeze(0)
    delta = onsets.shape[1] - hr.shape[1]
    onsets = onsets[:, :-delta]
    return torch.cat((onsets, hr), dim=0)


if __name__ == "__main__":
    pass
    # x = np.random.rand(1000)
    # features = differential_entropy(x)
    # print(f"features: {features} with shape {features.shape}")