from fusion.audio_processing import main as audio_main
from shared.constants import general_emotion_mapping
from datetime import datetime
import numpy as np
import random

def main(audio_model_path: str,
         audio_model_epoch: int,
         video_model_path: str,
         video_model_epoch: int,
         audio_frames: any,
         video_frames: any,
         live_demo: bool = False):
    # Audio processing
    audio_output = audio_main(model_path=audio_model_path, epoch=audio_model_epoch, audio_file=audio_frames, live_demo=live_demo)

    # Video processing
    video_frames = get_frames_duration(video_frames)   
    video_output = []
    # Note: The following code is a placeholder for the actual fusion logic
    for duration, frame in video_frames:
        emotions = compute_softmax([random.random() for _ in range(len(general_emotion_mapping.keys()))])
        video_output.append({'frame_duration': duration, 'emotions': emotions})
    ###

    compute_predictions(audio_output, video_output)

def get_frames_duration(video_frames):
    start_time = datetime.timestamp(video_frames[0][1])
    frame_duration = [(datetime.timestamp(frame[1]) - start_time, frame[0]) for frame in video_frames]
    return frame_duration

def compute_predictions(audio_output, video_output):
    # Compute the average of logits for each video frame within the corresponding audio window
    audio_start_times = [audio['start_time'] for audio in audio_output]
    audio_end_times = [audio['end_time'] for audio in audio_output]

    for i, audio_window in enumerate(zip(audio_start_times, audio_end_times)):
        window_start, window_end = audio_window
        video_logits_sum = np.zeros(len(general_emotion_mapping.keys()))
        video_frame_count = 0
        
        for video_frame in video_output:
            frame_duration = video_frame['frame_duration']
            if window_start <= frame_duration <= window_end:
                video_logits_sum += video_frame['emotions'] # Sum the logits of the video frame
                video_frame_count += 1
        video_logits_avg = video_logits_sum / video_frame_count
        
        audio_logits = audio_output[i]['logits'][0] # Get the logits of the audio window
        fused_logits = audio_logits + video_logits_avg # Sum the logits of the audio window and video frames
        
        fused_emotions = compute_softmax(fused_logits) # Convert logits to probabilities using softmax
        pred = np.argmax(fused_emotions, -1)
        emotion = general_emotion_mapping[pred.item()]
        
        print(f"Emotion detected from {window_start:.2f}s to {window_end:.2f}s: {emotion}")

def compute_softmax(logits):
    return np.exp(logits) / np.sum(np.exp(logits), axis=0)