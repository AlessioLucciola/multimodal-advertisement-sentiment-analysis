from config import *
from utils.utils import select_device, set_seed
import numpy as np
import torch
import json
import sys
import os
import cv2
from tqdm import tqdm
from shared.constants import ppg_emotion_mapping, CEAP_STD, CEAP_MEAN
from models.EmotionNetCEAP import EmotionNet, Encoder, Decoder
from packages.rppg_toolbox.main import extract_ppg_from_video
from utils.ppg_utils import wavelet_transform
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
    model, _ = get_model(model_path, device)
    model = load_test_model(model, model_path, epoch, device)
    emotions, timestamps = get_emotions_from_video(model, video_frames, device)
    ppg_output = [
            {'frame_duration': timestamp.item(), 
             'emotion_label': emotion.item(),
             'emotion_string': ppg_emotion_mapping[int(emotion.item())],}
            for emotion, timestamp in zip(emotions, timestamps) if timestamp != -1]

    return ppg_output

def get_emotions_from_video(model: EmotionNet, video_frames: str | np.ndarray, device) -> Tuple[torch.Tensor, torch.Tensor]:
    preds = torch.tensor([]).to(device)
    ppgs, timestamps = extract_ppg_from_video(vid_path=video_frames) #ppg shape: [num_chunks * num_splits, 100]
    ppgs = CEAP_MEAN + (ppgs - ppgs.view(-1).mean()) * (CEAP_STD / ppgs.view(-1).std())
    print(f"ppgs mean and std: {ppgs.view(-1).mean(), ppgs.view(-1).std()}")
    segment_preds = []
    memory, first_input = None, None
    for i, ppg in tqdm(enumerate(ppgs), desc="Inference..."):
        ppg = wavelet_transform(ppg.squeeze())
        ppg = torch.from_numpy(ppg).unsqueeze(1).to(device)
        #TODO: see what value to insert here as a starting token since it changes the prediction
        #TODO: maybe train the model using always the 0 as the starting input 
        trg = torch.tensor([0.0]).to(device)

        preds, memory = model(src=ppg, trg=trg, memory=memory, first_input=first_input)

        first_input = preds[-1].argmax(-1).float()
        segment_preds.append(preds)

    emotions = [segment.argmax(-1).squeeze() for segment in segment_preds]
    emotions = torch.cat(emotions, dim=0)
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
    
    input_dim = LENGTH // WAVELET_STEP if WT else 1
    output_dim = 3
    encoder_embedding_dim = LENGTH // WAVELET_STEP if WT else 1
    decoder_embedding_dim = LENGTH // WAVELET_STEP if WT else 1
    hidden_dim = (
        LSTM_HIDDEN
        if configurations is None
        else configurations["lstm_config"]["num_hidden"]
    )
    n_layers = (
        LSTM_LAYERS
        if configurations is None
        else configurations["lstm_config"]["num_layers"]
    )
    encoder_dropout = DROPOUT_P
    decoder_dropout = DROPOUT_P
    num_classes = EMOTION_NUM_CLASSES

    encoder = Encoder(
        input_dim,
        encoder_embedding_dim,
        hidden_dim,
        n_layers,
        encoder_dropout,
    )

    decoder = Decoder(
        output_dim,
        decoder_embedding_dim,
        hidden_dim,
        n_layers,
        decoder_dropout,
    )

    model = EmotionNet(encoder, decoder).to(device)
    return model, num_classes


def load_test_model(model, model_path, epoch, device):
    state_dict = torch.load(
        f"{PATH_TO_SAVE_RESULTS}/{model_path}/models/mi_project_{epoch}.pt",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model
