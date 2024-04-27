from shared.constants import general_emotion_mapping, merged_emotion_mapping
from fusion.audio_processing import main as audio_main
from fusion.video_processing import main as video_main
from datetime import datetime
from PIL import Image
import numpy as np
import random
from moviepy.editor import VideoFileClip
import os

def main(audio_model_path: str,
         audio_model_epoch: int,
         video_model_path: str,
         video_model_epoch: int,
         audio_frames: any,
         video_frames: any,
         live_demo: bool = True,
         use_positive_negative_labels = True
         ):
    pass

    if not live_demo: # Extract audio from the offline video file
        extract_audio_from_video(video_frames, audio_frames)

    # Audio processing
    audio_output = audio_main(model_path=audio_model_path, epoch=audio_model_epoch, audio_file=audio_frames, live_demo=live_demo)
    # Video processing
    video_output = video_main(model_path=video_model_path, video_frames=video_frames, epoch=video_model_epoch, live_demo=live_demo)
    video_output = get_frames_duration(video_output, live_demo)
    
    fused_emotion_lists = compute_fused_predictions(audio_output, video_output, use_positive_negative_labels) # Fusion logic in the time windows in which both audio and video are available
    remaining_video_frames = compute_remaining_video_predictions(fused_emotion_lists, video_output, use_positive_negative_labels) # Compute predictions for the remaining time windows only with video

    all_frames = sorted(fused_emotion_lists + remaining_video_frames, key=lambda x: x['start_time'])

    for f in all_frames:
       print(f)

    return all_frames

def extract_audio_from_video(video_frames, audio_frames):
    # Load the video file
    video_clip = VideoFileClip(video_frames)
    
    # Extract the audio
    audio_clip = video_clip.audio

    # Save the audio to a file
    audio_clip.write_audiofile(audio_frames)

    # Close the video file
    video_clip.close()

def get_frames_duration(video_frames, live_demo):
    if live_demo:
        start_time = datetime.timestamp(video_frames[0][1])
        frame_duration = [(datetime.timestamp(frame[1]) - start_time, frame[0]) for frame in video_frames]
    else:
        frame_duration = video_frames
    return frame_duration

def compute_fused_predictions(audio_output, video_output, use_positive_negative_labels):
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
        fused_logits = audio_logits + video_logits_avg # Sum the logits of the audio window and video frames
        
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
    video_output = substitute_frame_duration(video_output)
    remaining_video_frames = []

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
    
    fused_emotion_list = sorted(fused_emotion_list, key=lambda x: x['start_time'])
    for i, fused_frame in enumerate(fused_emotion_list):
        nearest_video_end_frame = min(remaining_video_frames, key=lambda x: abs(x['start_time'] - fused_frame['start_time']))
        nearest_video_end_frame_index = nearest_video_end_frame['index']
        remaining_video_frames[nearest_video_end_frame_index]['end_time'] = fused_frame['start_time']
        nearest_video_start_frame = min(remaining_video_frames, key=lambda x: abs(x['start_time'] - fused_frame['end_time']))
        nearest_video_start_frame_index = nearest_video_start_frame['index']
        remaining_video_frames[nearest_video_start_frame_index]['start_time'] = fused_frame['end_time']
    
    # Discard all the video frames that have a duration of 0
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

if __name__ == "__main__":
    audio_model_epoch = 500
    audio_model_path = os.path.join("AudioNetCT_2024-04-22_17-34-38")
    audio_file_path = os.path.join("data", "AUDIO", "test_audio_real.wav")

    video_model_epoch = 25
    video_model_path = os.path.join("VideoNet_vit-pretrained_2024-04-21_23-34-25")
    video_file_path = os.path.join("data", "VIDEO", "test_video.mp4")
    # video_file = [("<_io.BytesIO object at 0x000001BACE8C5D50>", datetime(2024, 4, 15, 17, 24, 42, 143608)),
    #               ("<_io.BytesIO object at 0x000001BA97A8C400>", datetime(2024, 4, 15, 17, 24, 42, 413783)),
    #               ("<_io.BytesIO object at 0x000001BA97A8C400>", datetime(2024, 4, 15, 17, 24, 42, 553515)),
    #               ("<_io.BytesIO object at 0x000001BA97A8C400>", datetime(2024, 4, 15, 17, 24, 42, 753758)),
    #               ("<_io.BytesIO object at 0x000001BA97A8C400>", datetime(2024, 4, 15, 17, 24, 42, 883772)),
    #               ("<_io.BytesIO object at 0x000001BA97A8C400>", datetime(2024, 4, 15, 17, 24, 43, 83427)),
    #               ("<_io.BytesIO object at 0x000001BA97A8C400>", datetime(2024, 4, 15, 17, 24, 43, 214003)),
    #               ("<_io.BytesIO object at 0x000001BA97836B60>", datetime(2024, 4, 15, 17, 24, 43, 485921)),
    #               ("<_io.BytesIO object at 0x000001BA97836B60>", datetime(2024, 4, 15, 17, 24, 43, 573439)),
    #               ("<_io.BytesIO object at 0x000001BA97836B60>", datetime(2024, 4, 15, 17, 24, 43, 773539)),
    #               ("<_io.BytesIO object at 0x000001BA97836B60>", datetime(2024, 4, 15, 17, 24, 43, 903753)),
    #               ("<_io.BytesIO object at 0x000001BA97836B60>", datetime(2024, 4, 15, 17, 24, 44, 103544)),
    #               ("<_io.BytesIO object at 0x000001BACF7E31A0>", datetime(2024, 4, 15, 17, 24, 44, 513430)),
    #               ("<_io.BytesIO object at 0x000001BACF7E31A0>", datetime(2024, 4, 15, 17, 24, 44, 653982)),
    #               ("<_io.BytesIO object at 0x000001BACF7E31A0>", datetime(2024, 4, 15, 17, 24, 44, 853632)),
    #               ("<_io.BytesIO object at 0x000001BACF7E31A0>", datetime(2024, 4, 15, 17, 24, 44, 984306)),
    #               ("<_io.BytesIO object at 0x000001BACF7E31A0>", datetime(2024, 4, 15, 17, 24, 45, 183510)),
    #               ("<_io.BytesIO object at 0x000001BA97A6E480>", datetime(2024, 4, 15, 17, 24, 45, 424064)),
    #               ("<_io.BytesIO object at 0x000001BA97A6E480>", datetime(2024, 4, 15, 17, 24, 45, 564150)),
    #               ("<_io.BytesIO object at 0x000001BA97A6E480>", datetime(2024, 4, 15, 17, 24, 45, 763909)),
    #               ("<_io.BytesIO object at 0x000001BA97A6E480>", datetime(2024, 4, 15, 17, 24, 45, 893495)),
    #               ("<_io.BytesIO object at 0x000001BA97A6E480>", datetime(2024, 4, 15, 17, 24, 46, 94230)),
    #               ("<_io.BytesIO object at 0x000001BACF4CF740>", datetime(2024, 4, 15, 17, 24, 46, 513780)),
    #               ("<_io.BytesIO object at 0x000001BACF4CF740>", datetime(2024, 4, 15, 17, 24, 46, 653800)),
    #               ("<_io.BytesIO object at 0x000001BACF4CF740>", datetime(2024, 4, 15, 17, 24, 46, 853409)),
    #               ("<_io.BytesIO object at 0x000001BACF4CF740>", datetime(2024, 4, 15, 17, 24, 46, 984371)),
    #               ("<_io.BytesIO object at 0x000001BACF4CF740>", datetime(2024, 4, 15, 17, 24, 47, 184249)),
    #               ("<_io.BytesIO object at 0x000001BACF8B4CC0>", datetime(2024, 4, 15, 17, 24, 47, 423469)),
    #               ("<_io.BytesIO object at 0x000001BACF8B4CC0>", datetime(2024, 4, 15, 17, 24, 47, 563986)),
    #               ("<_io.BytesIO object at 0x000001BACF8B4CC0>", datetime(2024, 4, 15, 17, 24, 47, 764301)),
    #               ("<_io.BytesIO object at 0x000001BACF8B4CC0>", datetime(2024, 4, 15, 17, 24, 47, 894020)),
    #               ("<_io.BytesIO object at 0x000001BACF8B4CC0>", datetime(2024, 4, 15, 17, 24, 48, 93631)),
    #               ("<_io.BytesIO object at 0x000001BA95E0FCE0>", datetime(2024, 4, 15, 17, 24, 48, 514107)),
    #               ("<_io.BytesIO object at 0x000001BA95E0FCE0>", datetime(2024, 4, 15, 17, 24, 48, 653489)),
    #               ("<_io.BytesIO object at 0x000001BA95E0FCE0>", datetime(2024, 4, 15, 17, 24, 48, 853745)),
    #               ("<_io.BytesIO object at 0x000001BA95E0FCE0>", datetime(2024, 4, 15, 17, 24, 48, 984133)),
    #               ("<_io.BytesIO object at 0x000001BA95E0FCE0>", datetime(2024, 4, 15, 17, 24, 49, 183707)),
    #               ("<_io.BytesIO object at 0x000001BACF4CFF10>", datetime(2024, 4, 15, 17, 24, 49, 433456)),
    #               ("<_io.BytesIO object at 0x000001BACF4CFF10>", datetime(2024, 4, 15, 17, 24, 49, 574117)),
    #               ("<_io.BytesIO object at 0x000001BACF4CFF10>", datetime(2024, 4, 15, 17, 24, 49, 774325)),
    #               ("<_io.BytesIO object at 0x000001BACF4CFF10>", datetime(2024, 4, 15, 17, 24, 49, 903608)),
    #               ("<_io.BytesIO object at 0x000001BACF4CFF10>", datetime(2024, 4, 15, 17, 24, 50, 103536)),
    #               ("<_io.BytesIO object at 0x000001BACF4B0F40>", datetime(2024, 4, 15, 17, 24, 50, 524222)),
    #               ("<_io.BytesIO object at 0x000001BACF4B0F40>", datetime(2024, 4, 15, 17, 24, 50, 663441)),
    #               ("<_io.BytesIO object at 0x000001BACF4B0F40>", datetime(2024, 4, 15, 17, 24, 50, 864047)),
    #               ("<_io.BytesIO object at 0x000001BACF4B0F40>", datetime(2024, 4, 15, 17, 24, 50, 993508)),
    #               ("<_io.BytesIO object at 0x000001BACF4B0F40>", datetime(2024, 4, 15, 17, 24, 51, 193912)),
    #               ("<_io.BytesIO object at 0x000001BACEAA8310>", datetime(2024, 4, 15, 17, 24, 51, 443588)),
    #               ("<_io.BytesIO object at 0x000001BACEAA8310>", datetime(2024, 4, 15, 17, 24, 51, 583734)),
    #               ("<_io.BytesIO object at 0x000001BACEAA8310>", datetime(2024, 4, 15, 17, 24, 51, 783970)),
    #               ("<_io.BytesIO object at 0x000001BACEAA8310>", datetime(2024, 4, 15, 17, 24, 51, 913902)),
    #               ("<_io.BytesIO object at 0x000001BACEAA8310>", datetime(2024, 4, 15, 17, 24, 52, 113753)),
    #               ("<_io.BytesIO object at 0x000001BACF4B1670>", datetime(2024, 4, 15, 17, 24, 52, 524013)),
    #               ("<_io.BytesIO object at 0x000001BACF4B1670>", datetime(2024, 4, 15, 17, 24, 52, 664265)),
    #               ("<_io.BytesIO object at 0x000001BACF4B1670>", datetime(2024, 4, 15, 17, 24, 52, 863739)),
    #               ("<_io.BytesIO object at 0x000001BACF4B1670>", datetime(2024, 4, 15, 17, 24, 52, 994163)),
    #               ("<_io.BytesIO object at 0x000001BACF4B1670>", datetime(2024, 4, 15, 17, 24, 53, 193439)),
    #               ("<_io.BytesIO object at 0x000001BACF8B5120>", datetime(2024, 4, 15, 17, 24, 53, 453992)),
    #               ("<_io.BytesIO object at 0x000001BACF8B5120>", datetime(2024, 4, 15, 17, 24, 53, 593698)),
    #               ("<_io.BytesIO object at 0x000001BACF8B5120>", datetime(2024, 4, 15, 17, 24, 53, 793735)),
    #               ("<_io.BytesIO object at 0x000001BACF8B5120>", datetime(2024, 4, 15, 17, 24, 53, 923930)),
    #               ("<_io.BytesIO object at 0x000001BACF8B5120>", datetime(2024, 4, 15, 17, 24, 54, 124118)),
    #               ("<_io.BytesIO object at 0x000001BACDCBA250>", datetime(2024, 4, 15, 17, 24, 54, 453746)),
    #               ("<_io.BytesIO object at 0x000001BACDCBA250>", datetime(2024, 4, 15, 17, 24, 54, 593697)),
    #               ("<_io.BytesIO object at 0x000001BACDCBA250>", datetime(2024, 4, 15, 17, 24, 54, 794176)),
    #               ("<_io.BytesIO object at 0x000001BACDCBA250>", datetime(2024, 4, 15, 17, 24, 54, 923437)),
    #               ("<_io.BytesIO object at 0x000001BACDCBA250>", datetime(2024, 4, 15, 17, 24, 55, 123434)),
    #               ("<_io.BytesIO object at 0x000001BA978E7D80>", datetime(2024, 4, 15, 17, 24, 55, 533935)),
    #               ("<_io.BytesIO object at 0x000001BA978E7D80>", datetime(2024, 4, 15, 17, 24, 55, 674109)),
    #               ("<_io.BytesIO object at 0x000001BA978E7D80>", datetime(2024, 4, 15, 17, 24, 55, 873618)),
    #               ("<_io.BytesIO object at 0x000001BA978E7D80>", datetime(2024, 4, 15, 17, 24, 56, 3925)),
    #               ("<_io.BytesIO object at 0x000001BA978E7D80>", datetime(2024, 4, 15, 17, 24, 56, 203719))]

    main(audio_model_path=audio_model_path,
         audio_model_epoch=audio_model_epoch,
         video_model_path=video_model_path,
         video_model_epoch=video_model_epoch,
         audio_frames=audio_file_path,
         video_frames=video_file_path,
         live_demo=False # Set to True if data arrives from the demo, false otherwise (offline test)
        )