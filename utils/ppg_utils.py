import torch
import numpy as np
from pyteap.signals.bvp import get_bvp_features
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample


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


def extract_ppg_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    tqdm.pandas()
    df["ppg_features"] = df["ppg"].progress_apply(
        lambda x: extract_ppg_features(x))
    return df
