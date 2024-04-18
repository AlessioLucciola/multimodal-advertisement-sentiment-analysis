from config import *
from utils.utils import select_device, set_seed
from shared.constants import merged_emotion_mapping
import numpy as np
import torch
import json
import sys
import os
import cv2  
from torchvision import transforms
from utils.video_utils import select_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main(model_path, video_file, epoch, live_demo=False):
    set_seed(RANDOM_SEED)
    device = select_device()
    type = model_path.split('_')[0]
    model, _ = get_model_and_dataloader(model_path, device, type)
    model = load_test_model(model, model_path, epoch, device)
    features_list = preprocess_frames(video_file, live_demo)
    video_processed_windows = []
    for feature in features_list:
        frame = feature['frame'].to(device)
        start_time = feature['start_time']
        end_time = feature['end_time']
        output = model(frame)
        pred = torch.argmax(output, -1).detach()
        emotion = merged_emotion_mapping[pred.item()]
        print(f"Emotion detected from {start_time:.2f}s to {end_time:.2f}s: {emotion}")

        video_processed_windows.append({
            "start_time": start_time,
            "end_time": end_time,
            "emotion_label": pred.item(),
            "emotion_string": emotion
        })

    return video_processed_windows

def preprocess_frames(video_file, live_demo, desired_length_seconds=VIDEO_DURATION):
    if live_demo:
        pass
    else:
        frames = extract_frames_from_video_file(file=video_file, desired_length_seconds=desired_length_seconds)
        transformations = transforms.Compose([
            transforms.ToTensor()])
        preprocessed_frames = []
        for frame in frames:
            frame['frame'] = transformations(frame['frame']).unsqueeze(0)
            preprocessed_frames.append({
                "frame": frame['frame'],
                "start_time": frame['start_time'],
                "end_time": frame['end_time']
            })
            
    return preprocessed_frames

def extract_frames_from_video_file(file, desired_length_seconds):
    cap = cv2.VideoCapture(file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    interval = int(frame_rate * desired_length_seconds)  # Extract a frame every 2 seconds

    frame_count = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frames every interval
        if frame_count % interval == 0:
            face_cascade = cv2.CascadeClassifier('./models/haarcascade/haarcascade_frontalface_default.xml')

            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.12, minNeighbors=9)
                    
            if len(faces) == 0: # No face detected
                faces = face_cascade.detectMultiScale(frame, scaleFactor=1.02, minNeighbors=9) # Try again with different parameters
                if len(faces) == 0: # Still no face detected
                    continue
            if len(faces) > 1: # More than one face detected
                # Choose the most prominent face
                face = max(faces, key=lambda x: x[2] * x[3])
                faces = [face]

            # Detect faces
            for (x, y, w, h) in faces:
                # Extract face from the frame
                face = frame[y:y+h, x:x+w]

                # Resize face
                face = cv2.resize(face, IMG_SIZE)

                # Plot face
                # cv2.imshow('frame', face)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                # Save frame
                frames.append({
                    "frame": face,
                    "start_time": frame_count / frame_rate,
                    "end_time": (frame_count + interval) / frame_rate
                })
        frame_count += 1
    cap.release()

    return frames
     

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
    num_classes = NUM_CLASSES if configurations is None else configurations["num_classes"]
    dropout_p = DROPOUT_P if configurations is None else configurations["dropout_p"]
    hidden_size = HIDDEN_SIZE if configurations is None else configurations["hidden_size"]
    model = select_model(model_path.split('_')[1], hidden_size, num_classes, dropout_p).to(device)

    return model, num_classes

if __name__ == "__main__":
    epoch = TEST_EPOCH
    model_path = os.path.join(PATH_MODEL_TO_TEST)
    # Offline video file, it you want to test with a live video stream use the demo instead
    video_file_path = os.path.join(VIDEO_DATASET_DIR, "test_video.mp4")
    main(model_path=model_path, video_file=video_file_path, epoch=epoch)