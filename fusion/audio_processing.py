from config import AUDIO_SAMPLE_RATE, AUDIO_OFFSET, AUDIO_DURATION, DROPOUT_P, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, NUM_MFCC, FRAME_LENGTH, HOP_LENGTH, PATH_TO_SAVE_RESULTS, NUM_CLASSES, RANDOM_SEED
from models.AudioNetCT import AudioNet_CNN_Transformers as AudioNetCT
from models.AudioNetCL import AudioNet_CNN_LSTM as AudioNetCL
from utils.audio_utils import extract_mfcc_features, extract_multiple_waveforms_from_audio_file, extract_multiple_waveforms_from_buffer, extract_waveform_from_audio_file, extract_features, detect_speech, extract_speech_segment_from_waveform
from utils.utils import upload_scaler, select_device, set_seed
from shared.constants import general_emotion_mapping, merged_emotion_mapping
import numpy as np
import torch
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main(model_path, audio_file, epoch, use_positive_negative_labels=True, live_demo=False):
    set_seed(RANDOM_SEED)
    device = select_device()
    type = model_path.split('_')[0]
    model, scaler, _ = get_model_and_dataloader(model_path, device, type)
    model = load_test_model(model, model_path, epoch, device)
    features_list = preprocess_audio_file(audio_file, scaler, live_demo)
    audio_processed_windows = []
    for feature in features_list:
        waveform = feature['waveform'].to(device)
        start_time = feature['start_time']
        end_time = feature['end_time']
        longest_voice_segment_start = feature['longest_voice_segment_start']
        longest_voice_segment_end = feature['longest_voice_segment_end']
        output = model(waveform)
        pred = torch.argmax(output, -1).detach()
        emotion = merged_emotion_mapping[pred.item()] if use_positive_negative_labels else general_emotion_mapping[pred.item()]
        #print(f"Emotion detected from {longest_voice_segment_start:.2f}s to {longest_voice_segment_end:.2f}s: {emotion}")
        audio_processed_windows.append({
            "start_time": start_time,
            "end_time": end_time,
            "longest_voice_segment_start": longest_voice_segment_start,
            "longest_voice_segment_end": longest_voice_segment_end,
            "longest_voice_segment_length": feature['longest_voice_segment_length'],
            "emotion_label": pred.item(),
            "emotion_string": emotion,
            "logits": torch.softmax(output, -1).cpu().detach().numpy()
        })
    
    audio_processed_windows = merge_overlapping_windows(audio_processed_windows)
    for emotion in audio_processed_windows:
        print(f"Emotion detected from {emotion['longest_voice_segment_start']:.2f}s to {emotion['longest_voice_segment_end']:.2f}s: {emotion['emotion_string']}")
    return audio_processed_windows

def preprocess_audio_file(audio_file, scaler, live_demo, desired_length_seconds=AUDIO_DURATION, desired_sample_rate=AUDIO_SAMPLE_RATE):
    if live_demo:
        segments = extract_multiple_waveforms_from_buffer(buffer=audio_file, desired_length_seconds=desired_length_seconds, desired_sample_rate=desired_sample_rate)
    else:
        segments = extract_multiple_waveforms_from_audio_file(file=audio_file, desired_length_seconds=desired_length_seconds, desired_sample_rate=desired_sample_rate)
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
        num_classes = NUM_CLASSES if configurations is None else configurations["num_classes"]
        num_mfcc = NUM_MFCC if configurations is None else configurations["num_mfcc"]
        dropout_p = DROPOUT_P if configurations is None else configurations["dropout_p"]
        model = AudioNetCT(
            num_classes=num_classes, num_mfcc=num_mfcc, dropout_p=dropout_p).to(device)
        scaler = upload_scaler(model_path)
    elif type == "AudioNetCL":
        num_classes = NUM_CLASSES if configurations is None else configurations["num_classes"]
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

def merge_overlapping_windows(data):
    merged_windows = []
    current_merge = None
    
    # Sort the data based on start_time
    sorted_data = sorted(data, key=lambda x: (x['emotion_label'], x['start_time']))
    
    # Iterate through the sorted data to merge overlapping windows
    for window in sorted_data:
        if current_merge is None:
            current_merge = window
            current_merge['logits_sum'] = window['logits']
            current_merge['num_windows'] = 1
        elif window['emotion_label'] == current_merge['emotion_label'] and window['start_time'] <= current_merge['end_time']:
            # Check if the current window completely includes the new window
            if window['end_time'] <= current_merge['end_time']:
                continue  # Skip the new window since it's completely included in the current one
            else:
                # Adjust the start time of the new window
                if window['start_time'] < current_merge['end_time']:
                    window['start_time'] = current_merge['end_time']
                # Merge with the current window
                current_merge['end_time'] = window['end_time']
                current_merge['longest_voice_segment_end'] = max(current_merge['longest_voice_segment_end'], window['longest_voice_segment_end'])
                current_merge['logits_sum'] += window['logits']
                current_merge['num_windows'] += 1
        else:
            # Check for overlaps with the previous window and adjust start time if necessary
            if current_merge['end_time'] > window['start_time']:
                window['start_time'] = current_merge['end_time']
            # Calculate the average logits
            current_merge['logits'] = current_merge['logits_sum'] / current_merge['num_windows']
            # Add the current merge to the list
            merged_windows.append(current_merge)
            current_merge = window
            current_merge['logits_sum'] = window['logits']
            current_merge['num_windows'] = 1
    
    # Check if there's a current merge to be added
    if current_merge is not None:
        # Calculate the average logits
        current_merge['logits'] = current_merge['logits_sum'] / current_merge['num_windows']
        merged_windows.append(current_merge)

    sorted_windows = sorted(merged_windows, key=lambda x: x['num_windows'], reverse=True)
    
    # Initialize a set to keep track of occupied time intervals
    occupied_intervals = set()
    
    for window in sorted_windows:
        start_time = window['longest_voice_segment_start']
        end_time = window['longest_voice_segment_end']
        
        # Check for overlaps with previous windows
        for interval in occupied_intervals:
            interval_start, interval_end = interval
            if start_time < interval_end and end_time > interval_start:
                # If overlap, adjust the start and end times
                start_time = max(start_time, interval_end)
                end_time = start_time + (window['longest_voice_segment_end'] - window['longest_voice_segment_start'])
        
        # Update the occupied_intervals set with the adjusted interval
        occupied_intervals.add((start_time, end_time))
        
        # Update the window dictionary with adjusted start and end times
        window['longest_voice_segment_start'] = start_time
        window['longest_voice_segment_end'] = end_time
    
    return sorted_windows

