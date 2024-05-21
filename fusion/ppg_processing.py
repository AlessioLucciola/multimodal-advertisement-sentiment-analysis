from config import *
from utils.utils import select_device, set_seed
import numpy as np
import torch
import json
import sys
import os
from tqdm import tqdm
from shared.constants import ppg_emotion_mapping, SCALED_DEAP_STD, SCALED_DEAP_MEAN
from models.EmotionNetDEAP import EmotionNet
from packages.rppg_toolbox.main import extract_ppg_from_video
from utils.ppg_utils import fft, detrend, bandpass_filter, moving_average_filter
from typing import Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main(model_path: str, 
         video_frames: str | np.ndarray, 
         epoch: int, 
         use_positive_negative_labels:bool = True, 
         live_demo:bool=False):
    if not use_positive_negative_labels:
        raise NotImplementedError("Only negative, neutral, positive labels supported for PPG signal, please set use_positive_negative_labels to True")
    set_seed(RANDOM_SEED)
    device = select_device()
    model = get_model(model_path, device)
    model = load_test_model(model, model_path, epoch, device)
    emotions, timestamps = get_emotions_from_video(model, video_frames, device)
    ppg_output = [
            {'frame_duration': timestamp.item(), 
             'emotion_label': emotion.item(),
             'emotion_string': ppg_emotion_mapping[int(emotion.item())],}
            for emotion, timestamp in zip(emotions, timestamps) if timestamp != -1]

    return ppg_output

def preprocess_ppg(ppgs):
    preprocessed = []
    for ppg in ppgs:
        ppg = detrend(ppg)
        ppg = bandpass_filter(ppg)
        ppg = moving_average_filter(ppg)
        ppg = (ppg - min(ppg) / (max(ppg) - min(ppg)))
        ppg = (ppg - SCALED_DEAP_MEAN) / SCALED_DEAP_STD 
        ppg = fft(ppg)
        if ppg.shape[-1] != 128:
            ppg = ppg[:, :128]

        print(f"single ppg shape preprocessing is: {ppg.shape}")
        preprocessed.append(ppg)
    return preprocessed

def get_emotions_from_video(model: EmotionNet, video_frames: str | np.ndarray, device) -> Tuple[torch.Tensor, torch.Tensor]:
    #ppg shape: [num_chunks * num_splits, 100]
    ppgs, timestamps = extract_ppg_from_video(vid_path=video_frames) 
    ppgs = preprocess_ppg(ppgs)
    segment_preds = []
    for i, ppg in tqdm(enumerate(ppgs), desc="Inference..."):
        ppg = torch.from_numpy(ppg).float().to(device).unsqueeze(0)
        output =  model(ppg)
        
        repeated_output = output.argmax(-1).squeeze().repeat(ppg.shape[-1])
        print(f"repeated_output shape: {repeated_output.shape}")
        segment_preds.append(repeated_output)

    emotions = torch.cat(segment_preds, dim=0)
    print(f"emotions are: {emotions} with shape {emotions.shape}")
    timestamps = torch.cat([torch.tensor(ts) for ts in timestamps], dim=0)
    print(f"timestamps are: {timestamps} with shape {timestamps.shape}")
    return emotions, timestamps

def get_model(model_path, device):
    # Load configuration
    conf_path = PATH_TO_SAVE_RESULTS + f"/{model_path}/configurations.json"
    configurations = None
    if os.path.exists(conf_path):
        print(
            "--Model-- Old configurations found. Using those configurations for the test."
        )
        with open(conf_path, "r") as json_file:
            configurations = json.load(json_file)
    else:
        print(
            "--Model-- Old configurations NOT found. Using configurations in the config for test."
        )
    
    model = EmotionNet(dropout_p=configurations["dropout_p"] if configurations else DROPOUT_P).to(device)
    return model


def load_test_model(model, model_path, epoch, device):
    state_dict = torch.load(
        f"{PATH_TO_SAVE_RESULTS}/{model_path}/models/mi_project_{epoch}.pt",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model
