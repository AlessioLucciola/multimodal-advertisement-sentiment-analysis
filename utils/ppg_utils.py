import pywt
from typing import Callable
import torch
import numpy as np
from pyteap.signals.bvp import get_bvp_features
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
from biosppy.signals import bvp
from torch import nn

from config import LENGTH


def extract_ppg_features(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    x = np.concatenate((x, np.zeros(200)), axis=0)
    SR = 100
    x = resample(x, int(len(x) * SR / 100))
    features = torch.zeros((1, 17))
    return features
    if len(x.shape) == 1:
        if type(x) == torch.Tensor:
            feature = get_bvp_features(x.cpu().numpy(), sr=SR)
        elif type(x) == np.ndarray:
            feature = get_bvp_features(x, sr=SR)
        else:
            raise ValueError(
                "Invalid type of x. Must be torch.Tensor or np.ndarray.")
        return torch.tensor(feature).unsqueeze(0)
    for i in range(x.shape[0]):
        if type(x) == torch.Tensor:
            feature = get_bvp_features(x[i].cpu().numpy(), sr=SR)
        elif type(x) == np.ndarray:
            feature = get_bvp_features(x[i], sr=SR)
        else:
            raise ValueError(
                "Invalid type of x. Must be torch.Tensor or np.ndarray.")
        features[i] = torch.tensor(feature)
    return features


def statistical_features(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    mean = torch.mean(x, dim=1).unsqueeze(0)
    std = torch.std(x, dim=1).unsqueeze(0)
    skewness = torch.tensor(
        [pd.Series(x[0]).skew()]).unsqueeze(0)
    kurtosis = torch.tensor(
        [pd.Series(x[0]).kurtosis()]).unsqueeze(0)
    onsets, hr = onsets_and_hr()
    onsets = onsets.mean()
    hr = hr.mean()
    return torch.cat((mean, std, skewness, kurtosis, onsets, hr), dim=0)


def wavelet_transform(x):
    scale_step = 100
    # return np.zeros((2000 // scale_step,  2000))  # TODO: for debug

    scales = np.arange(1, len(x) + 1, scale_step)
    coef, freqs = pywt.cwt(x, scales, 'morl')
    # print(f"coef: {coef.shape}")
    return coef


def differential_entropy(x):
    # Estimate the probability density function of x by computing a histogram
    hist, bin_edges = np.histogram(x, bins='auto', density=True)
    # Compute the differential entropy
    diff_entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))

    return diff_entropy


def fft(x):
    spectrum = np.abs(np.fft.rfft(x))
    # normalized_spectrum = spectrum / sum(spectrum)
    # normalized_frequencies = np.linspace(0, 1, len(spectrum))
    # centroid = sum(normalized_frequencies * normalized_spectrum)
    print(f'spectrum: {spectrum.shape}')
    return spectrum


def onsets_and_hr(x):
    features = bvp.bvp(x, 100, show=False)
    onsets = torch.tensor(features["onsets"]).unsqueeze(0)
    hr = torch.tensor(features["heart_rate"]).unsqueeze(0)
    delta = onsets.shape[1] - hr.shape[1]
    onsets = onsets[:, :-delta]
    return torch.cat((onsets, hr), dim=0)


if __name__ == "__main__":
    x = np.random.rand(1000)
    features = differential_entropy(x)
    print(f"features: {features} with shape {features.shape}")
