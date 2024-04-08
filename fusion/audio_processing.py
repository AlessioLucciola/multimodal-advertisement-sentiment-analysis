from config import AUDIO_SAMPLE_RATE, AUDIO_OFFSET, AUDIO_DURATION, DROPOUT_P, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, NUM_MFCC, FRAME_LENGTH, HOP_LENGTH, PATH_TO_SAVE_RESULTS, RAVDESS_NUM_CLASSES, RANDOM_SEED
from models.AudioNetCT import AudioNet_CNN_Transformers as AudioNetCT
from models.AudioNetCL import AudioNet_CNN_LSTM as AudioNetCL
from utils.audio_utils import extract_mfcc_features, extract_multiple_waveforms_from_audio_file, extract_waveform_from_audio_file, extract_features, detect_speech, extract_speech_segment_from_waveform
from utils.utils import upload_scaler, select_device, set_seed
from shared.constants import general_emotion_mapping
import numpy as np
import torch
import json
import os

def main(model_path, audio_file_path, epoch):
    set_seed(RANDOM_SEED)
    device = select_device()
    type = model_path.split('_')[0]
    model, scaler, _ = get_model_and_dataloader(model_path, device, type)
    model = load_test_model(model, model_path, epoch, device)
    features_list = preprocess_audio_file(audio_file_path, scaler)
    for feature in features_list:
        waveform = feature['waveform'].to(device)
        start_time = feature['start_time']
        end_time = feature['end_time']
        longest_voice_segment_start = feature['longest_voice_segment_start']
        longest_voice_segment_end = feature['longest_voice_segment_end']
        output = model(waveform)
        pred = torch.argmax(output, -1).detach()
        emotion = general_emotion_mapping[pred.item()]
        print(f"Emotion detected from {longest_voice_segment_start:.2f}s to {longest_voice_segment_end:.2f}s: {emotion}")

def preprocess_audio_file(audio_file_path, scaler, desired_length_seconds=AUDIO_DURATION, desired_sample_rate=AUDIO_SAMPLE_RATE):
    segments = extract_multiple_waveforms_from_audio_file(file=audio_file_path, desired_length_seconds=desired_length_seconds, desired_sample_rate=desired_sample_rate)
    preprocessed_segments = []
    for segment in segments:
        speech_segments = detect_speech(waveform=segment['waveform'], start_time=segment['start_time'], end_time=segment['end_time'], sr=AUDIO_SAMPLE_RATE)
        if len(speech_segments) != 0:
            segment['waveform'], segment['longest_voice_segment_start'], segment['longest_voice_segment_end'], segment['longest_voice_segment_length'] = extract_speech_segment_from_waveform(waveform=segment['waveform'], start_time=segment['start_time'], end_time=segment['end_time'], speech_segments=speech_segments, sr=AUDIO_SAMPLE_RATE)
            preprocessed_segments.append(segment)
    for segment in preprocessed_segments:
        waveform = segment['waveform']
        features = extract_mfcc_features(waveform, sample_rate=AUDIO_SAMPLE_RATE, n_mfcc=NUM_MFCC, n_fft=1024, win_length=512, n_mels=128, window='hamming')
        features = scale_waveform(features, scaler)
        features = torch.from_numpy(np.expand_dims(np.expand_dims(features, axis=0), axis=0)).float()
        segment['waveform'] = features
    return preprocessed_segments

def extract_audio_features(audio_file_path, scaler):
    # Load the audio file
    waveform = extract_waveform_from_audio_file(file=audio_file_path, desired_length_seconds=AUDIO_DURATION, offset=AUDIO_OFFSET, desired_sample_rate=AUDIO_SAMPLE_RATE)
    # Extract features from the audio file
    # features = extract_features(waveform=waveform, sample_rate=AUDIO_SAMPLE_RATE, n_mfcc=NUM_MFCC, n_fft=1024, win_length=512, n_mels=128, window='hamming', frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    features = extract_mfcc_features(waveform, sample_rate=AUDIO_SAMPLE_RATE, n_mfcc=NUM_MFCC, n_fft=1024, win_length=512, n_mels=128, window='hamming')
    # Scale the waveform
    features = scale_waveform(features, scaler)
    features = np.expand_dims(np.expand_dims(features, axis=0), axis=0) # Add channel dimension to get a 4D tensor suitable for CNN
    return features

def scale_waveform(waveform, scaler):
    return scaler.transform(waveform.reshape(-1, waveform.shape[-1])).reshape(waveform.shape)

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
    if type == "AudioNetCT":
        num_classes = RAVDESS_NUM_CLASSES if configurations is None else configurations["num_classes"]
        num_mfcc = NUM_MFCC if configurations is None else configurations["num_mfcc"]
        dropout_p = DROPOUT_P if configurations is None else configurations["dropout_p"]
        model = AudioNetCT(
            num_classes=num_classes, num_mfcc=num_mfcc, dropout_p=dropout_p).to(device)
        scaler = upload_scaler(model_path)
    elif type == "AudioNetCL":
        num_classes = RAVDESS_NUM_CLASSES if configurations is None else configurations["num_classes"]
        num_mfcc = NUM_MFCC if configurations is None else configurations["num_mfcc"]
        lstm_hidden_size = LSTM_HIDDEN_SIZE if configurations is None else configurations["lstm_hidden_size"]
        lstm_num_layers = LSTM_NUM_LAYERS if configurations is None else configurations["lstm_num_layers"]
        dropout_p = DROPOUT_P if configurations is None else configurations["dropout_p"]
        model = AudioNetCL(
            num_classes=num_classes, num_mfcc=num_mfcc, num_layers=lstm_num_layers, hidden_size=lstm_hidden_size, dropout_p=dropout_p).to(device)
        scaler = upload_scaler(model_path)
    else:
        raise ValueError(f"Unknown architecture {type}")
    return model, scaler, num_classes

if __name__ == "__main__":
    epoch = 484
    model_path = os.path.join("AudioNetCT_2024-04-08_09-33-56")
    audio_file_path = os.path.join("data", "AUDIO", "test_audio_real.wav")
    main(model_path=model_path, audio_file_path=audio_file_path, epoch=epoch)