from shared.constants import general_emotion_mapping, merged_emotion_mapping
from fusion.audio_processing import main as audio_main
from fusion.video_processing import main as video_main
from fusion.ppg_processing import main as ppg_main
import numpy as np
import os
from utils.audio_utils import extract_audio_from_video
from typing import Any

def main(audio_model_path: str,
         audio_model_epoch: int,
         video_model_path: str,
         video_model_epoch: int,
         ppg_model_path: str,
         ppg_model_epoch: int,
         audio_frames: Any,
         video_frames: Any,
         live_demo: bool = True,
         use_positive_negative_labels = True,
         get_audio_from_video = True,
         audio_importance = 0.60,
         ):
    """
    It returns a tuple where the first element contains the audio only/ video only/ audio-video fused windows; and the second element the ppg_windows, if ppg is used, else None
    """
    if not live_demo and get_audio_from_video: # Extract audio from the offline video file
        if not os.path.exists(os.path.join("data", "AUDIO")):
            os.makedirs(os.path.join("data", "AUDIO"))
        extract_audio_from_video(video_file=video_frames, audio_path=audio_frames)

    # PPG processing
    ppg_output = ppg_main(model_path=ppg_model_path, video_frames=video_frames, epoch=ppg_model_epoch, live_demo=live_demo)
    # Audio processing
    audio_output = audio_main(model_path=audio_model_path, epoch=audio_model_epoch, audio_file=audio_frames, live_demo=live_demo)
    # Video processing
    video_output = video_main(model_path=video_model_path, video_frames=video_frames, epoch=video_model_epoch, live_demo=live_demo)
    
    ppg_windows_list = None
    if len(ppg_output):
        ppg_windows_list = create_ppg_windows(ppg_output)
    # return None, ppg_windows_list #TODO: early return just for debug
    
    if len(audio_output) == 0 and len(video_output) == 0:
        return None, ppg_windows_list
    elif len(video_output) == 0:
        audio_windows_list = []
        for audio in audio_output:
            audio_windows_list.append({
                "start_time": audio['longest_voice_segment_start'],
                "end_time": audio['longest_voice_segment_end'],
                "emotion_label": audio['emotion_label'],
                "emotion_string": audio['emotion_string'],
                "window_type": "audio"
            })
        return audio_windows_list, ppg_windows_list
    elif len(audio_output) == 0:
        return create_video_windows(video_output), ppg_windows_list
    else:
        fused_emotion_lists = compute_fused_predictions(audio_output, video_output, use_positive_negative_labels, audio_importance=audio_importance) # Fusion logic in the time windows in which both audio and video are available
        remaining_video_frames = compute_remaining_video_predictions(fused_emotion_lists, video_output, use_positive_negative_labels) # Compute predictions for the remaining time windows only with video

        all_frames = sorted(fused_emotion_lists + remaining_video_frames, key=lambda x: x['start_time'])

        return all_frames, ppg_windows_list

def compute_fused_predictions(audio_output, video_output, use_positive_negative_labels, audio_importance):
    # Compute the average of logits for each video frame within the corresponding audio window
    audio_start_times = [audio['longest_voice_segment_start'] for audio in audio_output]
    audio_end_times = [audio['longest_voice_segment_end'] for audio in audio_output]

    fused_emotions_list = []
    for i, audio_window in enumerate(zip(audio_start_times, audio_end_times)):
        window_start, window_end = audio_window
        video_logits_sum = np.zeros(len(merged_emotion_mapping.keys() if use_positive_negative_labels else general_emotion_mapping.keys()))
        video_frame_count = 0
        
        for video_frame in video_output:
            frame_duration = video_frame['frame_duration']
            if window_start <= frame_duration <= window_end:
                video_logits_sum += video_frame['logits'][0] # Sum the logits of the video frame
                video_frame_count += 1
        video_logits_avg = video_logits_sum / video_frame_count
        
        audio_logits = audio_output[i]['logits_sum'][0] # Get the logits of the audio window
        audio_logits = compute_softmax(audio_logits) # Convert logits to probabilities using softmax
        fused_logits = (audio_logits * audio_importance) + (video_logits_avg * (1-audio_importance)) # Sum the logits of the audio window and video frames
        
        fused_emotions = fused_logits / 2 # Average the audio and video logits
        pred = np.argmax(fused_emotions, -1)
        emotion = merged_emotion_mapping[pred.item()] if use_positive_negative_labels else general_emotion_mapping[pred.item()]

        fused_emotions_list.append({
            "output": fused_logits,
            "start_time": audio_output[i]["longest_voice_segment_start"],
            "end_time": audio_output[i]["longest_voice_segment_end"],
            "emotion_label": pred.item(),
            "emotion_string": emotion,
            "window_type": "fusion"
        })
        
    return fused_emotions_list

def compute_remaining_video_predictions(fused_emotion_list, video_output, use_positive_negative_labels):
    # Substitute the video frames timestamps to create video windows with a duration of max 0.30 seconds
    video_output = substitute_frame_duration(video_output)
    remaining_video_frames = []

    # Check if the video frames intersect with the fused emotion windows, if not add them immediatly to the remaining video frames.
    # Otherwise discard them since they are already considered in the fused emotion windows
    for frame_video in video_output:
        intersected = False
        for frame_fused in fused_emotion_list:
            if (frame_fused['start_time'] <= frame_video['start_time'] <= frame_fused['end_time'] or
                frame_fused['start_time'] <= frame_video['end_time'] <= frame_fused['end_time'] or
                (frame_video['start_time'] <= frame_fused['start_time'] and frame_video['end_time'] >= frame_fused['end_time'])):
                intersected = True
                break
        
        if not intersected:
            remaining_video_frames.append(frame_video)

    # Add index to each frame to keep track of the video frames in the next step
    for i, d in enumerate(remaining_video_frames):
        d['index'] = i
    
    # For the remaining video frames, find the nearest fused emotion window and update the start and end time of the video frame
    '''
    fused_emotion_list = sorted(fused_emotion_list, key=lambda x: x['start_time'])
    for i, fused_frame in enumerate(fused_emotion_list):
        nearest_video_end_frame = min(remaining_video_frames, key=lambda x: abs(x['start_time'] - fused_frame['start_time']))
        nearest_video_end_frame_index = nearest_video_end_frame['index']
        remaining_video_frames[nearest_video_end_frame_index]['end_time'] = fused_frame['start_time']
        nearest_video_start_frame = min(remaining_video_frames, key=lambda x: abs(x['start_time'] - fused_frame['end_time']))
        nearest_video_start_frame_index = nearest_video_start_frame['index']
        remaining_video_frames[nearest_video_start_frame_index]['start_time'] = fused_frame['end_time']
    '''
    # Discard all the video frames that have a duration of 0 (if any)
    remaining_video_frames = [frame for frame in remaining_video_frames if frame['start_time'] != frame['end_time']]

    # Compute the predictions for the remaining video frames, remove the index and add the window type
    for frame in remaining_video_frames:
        pred = np.argmax(frame['logits'], -1)
        frame['emotion_label'] = pred.item() # Get the emotion label
        frame['emotion_string'] = merged_emotion_mapping[pred.item()] if use_positive_negative_labels else general_emotion_mapping[pred.item()] # Get the emotion string
        del frame['index'] # Remove the index
        frame['window_type'] = 'video' # Add the window type
    
    return remaining_video_frames

def substitute_frame_duration(video_output):
    for i in range(len(video_output)):
        if i == 0:
            video_output[i]['start_time'] = 0.0
            video_output[i]['end_time'] = video_output[i]['frame_duration']
        else:
            video_output[i]['start_time'] = video_output[i - 1]['end_time']
            video_output[i]['end_time'] = video_output[i]['frame_duration']
        del video_output[i]['frame_duration']
    return video_output

def compute_softmax(logits):
    return np.exp(logits) / np.sum(np.exp(logits), axis=0)

def create_video_windows(video_output):
    video_windows_list = []
    for i, frame in enumerate(video_output):
        if i == 0:
            start_time = 0.0
        else:
            start_time = video_windows_list[i - 1]['end_time']
        end_time = frame['frame_duration']
        video_windows_list.append({
            "start_time": start_time,
            "end_time": end_time,
            "emotion_label": frame['emotion_label'],
            "emotion_string": frame['emotion_string'],
            "window_type": "video"
        })
    return video_windows_list

def create_ppg_windows(ppg_output):
    ppg_windows_list = []
    for i, frame in enumerate(ppg_output):
        if i == 0:
            start_time = 0.0
        else:
            start_time = ppg_windows_list[i - 1]['end_time']
        end_time = frame['frame_duration']
        ppg_windows_list.append({
            "start_time": start_time,
            "end_time": end_time,
            "emotion_label": frame['emotion_label'],
            "emotion_string": frame['emotion_string'],
            "window_type": "ppg"
        })
    return ppg_windows_list

if __name__ == "__main__":
    audio_model_epoch = 215
    audio_model_path = os.path.join("AudioNetCT_2024-05-05_11-51-20")
    audio_file_path = os.path.join("data", "AUDIO", "test_audio_real.wav") # Necessary only if get_audio_from_video is false

    video_model_epoch = "best"
    video_model_path = os.path.join("VideoNet_vit-pretrained_2024-04-21_23-34-25")
    video_file_path = os.path.join("data", "VIDEO", "test_video_real.mp4")
    
    main(audio_model_path=audio_model_path,
         audio_model_epoch=audio_model_epoch,
         video_model_path=video_model_path,
         video_model_epoch=video_model_epoch,
         audio_frames=audio_file_path,
         video_frames=video_file_path,
         live_demo=False, # Set to True if data arrives from the demo, false otherwise (offline test)
         get_audio_from_video=True # Set to True if you want to extract the audio from the video, otherwise you must specify the path of the audio file
        )
