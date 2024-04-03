import torch
import numpy as np
from pyteap.signals.bvp import get_bvp_features
import pandas as pd
from tqdm import tqdm


def extract_ppg_features(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    features = torch.zeros((x.shape[0], 17))
    if len(x.shape) == 1:
        if type(x) == torch.Tensor:
            feature = get_bvp_features(x.cpu().numpy(), sr=90)
        elif type(x) == np.ndarray:
            feature = get_bvp_features(x, sr=90)
        else:
            raise ValueError(
                "Invalid type of x. Must be torch.Tensor or np.ndarray.")
        return torch.tensor(feature)
    for i in range(x.shape[0]):
        if type(x) == torch.Tensor:
            feature = get_bvp_features(x[i].cpu().numpy(), sr=90)
        elif type(x) == np.ndarray:
            feature = get_bvp_features(x[i], sr=90)
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
