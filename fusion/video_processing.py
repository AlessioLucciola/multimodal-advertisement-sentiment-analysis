from config import PATH_MODEL_RESULTS, MODEL_EPOCH, LIVE_TEST, AUDIO_SAMPLE_RATE, VIDEO_SAMPLE_RATE, VIDEO_OVERLAPPING_SECONDS, AUDIO_OFFSET, AUDIO_DURATION, VIDEO_DURATION, DROPOUT_P, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, NUM_MFCC, FRAME_LENGTH, HOP_LENGTH, PATH_TO_SAVE_RESULTS, RAVDESS_NUM_CLASSES, RANDOM_SEED, METADATA_CSV, VIDEO_NUM_CLASSES, BATCH_SIZE, LIMIT, BALANCE_DATASET, USE_DEFAULT_SPLIT, MODEL_NAME, HIDDEN_SIZE, DROPOUT_P, RESUME_TRAINING, DATASET_NAME, APPLY_TRANSFORMATIONS, DF_SPLITTING
from models.AudioNetCT import AudioNet_CNN_Transformers as AudioNetCT
from models.AudioNetCL import AudioNet_CNN_LSTM as AudioNetCL
from utils.audio_utils import extract_mfcc_features, extract_multiple_waveforms_from_audio_file, extract_waveform_from_audio_file, extract_features, detect_speech, extract_speech_segment_from_waveform
from utils.utils import upload_scaler, select_device, set_seed
from shared.constants import general_emotion_mapping
import numpy as np
import torch
import json
import os
import cv2
from PIL import Image
from torchvision import transforms
from utils.video_utils import select_model
from config import *

def main(model_path, video_file_path, epoch):
    set_seed(RANDOM_SEED)
    device = select_device()
    type = model_path.split('_')[0]
    model, scaler, _ = get_model_and_dataloader(model_path, device, type)
    model = load_test_model(model, model_path, epoch, device)

    test_on_video_file(model, video_file_path, device)

def test_on_video_file(model, video_file_path, device):
    features_list = preprocess_video_file(video_file_path)
    for feature in features_list:
        clip = feature['segment'].to(device)
        start_time = feature['start_time']
        end_time = feature['end_time']
        output = model(clip)
        pred = torch.argmax(output, -1).detach()
        emotion = general_emotion_mapping[pred.item()]
        print(f"Emotion detected from {start_time:.2f}s to {end_time:.2f}s: {emotion}")

def preprocess_video_file(video_file_path, desired_length_seconds=VIDEO_DURATION):
    segments = extract_multiple_frames_from_video_file(file=video_file_path, desired_length_seconds=desired_length_seconds, desired_sample_rate=VIDEO_SAMPLE_RATE)
    preprocessed_segments = []
    for segment in segments:
        segment['segment'] = torch.from_numpy(np.expand_dims(np.expand_dims(segment['segment'], axis=0), axis=0)).float()
        preprocessed_segments.append(segment)
    return preprocessed_segments

def extract_multiple_frames_from_video_file(file, desired_length_seconds, desired_sample_rate, overlap_seconds=VIDEO_OVERLAPPING_SECONDS):
    # Load the entire video file
    video = cv2.VideoCapture(file)

    # Calculate the number of samples corresponding to the desired length and overlap
    desired_length_samples = int(desired_length_seconds * desired_sample_rate)
    overlap_samples = int(overlap_seconds * desired_sample_rate)

    # Determine the step size for sliding the window
    step_size = desired_length_samples - overlap_samples

    # Determine the number of segments needed
    num_segments = int(video.get(cv2.CAP_PROP_FRAME_COUNT) / step_size)

    segments = []

    # Extract overlapping segments
    for i in range(num_segments):
        start_frame = i * step_size
        end_frame = start_frame + desired_length_samples

        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        while video.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
            ret, frame = video.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        frames = np.array(frames)
        segment = np.mean(frames, axis=0)
        start_time = start_frame / desired_sample_rate
        end_time = end_frame / desired_sample_rate
        segments.append({
            'segment': segment,
            'start_time': start_time,
            'end_time': end_time
        })

    return segments

def load_test_model(model, model_path, epoch, device):
    state_dict = torch.load(
        f"{PATH_TO_SAVE_RESULTS}/{model_path}/models/mi_project_{epoch}.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_model_and_dataloader(model_path, device, type):
    # Load configuration
    conf_path = PATH_TO_SAVE_RESULTS + f"/{model_path}/configurations.json"
    configurations = None
    if os.path.exists(conf_path):
        print(
            "--Model-- Old configurations found. Using those configurations for the test.")
        with open(conf_path, 'r') as json_file:
            configurations = json.load(json_file)
    else:
        print("--Model-- Old configurations NOT found. Using configurations in the config for test.")

    # Load model
    model = None
    scaler = None
    
    num_classes = VIDEO_NUM_CLASSES if configurations is None else configurations["num_classes"]
    dropout_p = DROPOUT_P if configurations is None else configurations["dropout_p"]

    model = select_model(MODEL_NAME, HIDDEN_SIZE, num_classes, dropout_p).to(device)
    
    scaler = None

    return model, scaler, num_classes

if __name__ == "__main__":
    epoch = TEST_EPOCH
    model_path = os.path.join(PATH_MODEL_TO_TEST)
    video_file_path = os.path.join("data", "VIDEO", "test_video.mp4")
    main(model_path=model_path, video_file_path=video_file_path, epoch=epoch)