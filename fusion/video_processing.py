from config import *
from utils.utils import select_device, set_seed
import numpy as np
import torch
import json
import sys
import os
from PIL import Image
from datetime import datetime
from utils.video_utils import select_model
import cv2
from torchvision import transforms
from shared.constants import general_emotion_mapping, merged_emotion_mapping

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main(model_path, video_frames, epoch, use_positive_negative_labels=True, live_demo=False):
    set_seed(RANDOM_SEED)
    device = select_device()
    type = model_path.split('_')[1]
    model, _ = get_model_and_dataloader(model_path, device, type)
    model = load_test_model(model, model_path, epoch, device)
    video_output = []
    current_datetime = datetime.now()
    current_datetime_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    video_path = os.path.join(DEMO_DIR, "video_files", str(current_datetime_str))
    if not os.path.exists(video_path):
        os.makedirs(video_path)

    # Load the face cascade
    face_cascade = cv2.CascadeClassifier('./models/haarcascade/haarcascade_frontalface_default.xml')

    # Define the transformation
    val_transform = transforms.Compose([transforms.ToTensor()])
    
    if live_demo:
        frames_extracted = 0
        video_starting_time = None
        for i, (frame, frame_duration) in enumerate(video_frames):
            if i == 0:
                video_starting_time = datetime.timestamp(frame_duration)
            frame_duration = datetime.timestamp(frame_duration) - video_starting_time
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.12, minNeighbors=9)

            if len(faces) == 0: # No face detected
                faces = face_cascade.detectMultiScale(frame, scaleFactor=1.02, minNeighbors=9) # Try again with different parameters
            if len(faces) == 0: # Still no face detected
                continue
            if len(faces) > 1: # More than one face detected
                # Choose the most prominent face
                face = max(faces, key=lambda x: x[2] * x[3])
                faces = [face]

            # Crop the face
            for (x, y, w, h) in faces:
                # Extract face from the frame
                face = frame[y:y+h, x:x+w]

                # Save the frame to the disk
                cv2.imwrite(os.path.join(video_path, f"{current_datetime_str}_{frames_extracted}.jpg"), face)
    
                # Resize face
                face = cv2.resize(face, IMG_SIZE)
                img = Image.fromarray(face)
                img = val_transform(img).unsqueeze(0)
                img = img.to(device)

                # Get prediction from model
                output = model(img)

                pred = torch.argmax(output, -1).detach()
                emotion = merged_emotion_mapping[pred.item()] if use_positive_negative_labels else general_emotion_mapping[pred.item()]

                # Return frame duration (float in seconds) and output (numpy array)
                video_output.append({
                    'frame_duration': frame_duration, 
                    'emotion_label': pred.item(),
                    'emotion_string': emotion,
                    'logits': torch.softmax(output, -1).cpu().detach().numpy()
                    })
    
            frames_extracted += 1
    else: 
        # Offline video file
        cap = cv2.VideoCapture(video_frames)

        # Variable to keep track of extracted frames
        frames_extracted = 0

        # Read the video file
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Compute frame duration (float in seconds)
            frame_duration = cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FPS)

            if int(frame_duration) % 2 == 0:
                # Detect face in the frame  
                faces = face_cascade.detectMultiScale(frame, scaleFactor=1.12, minNeighbors=9)

                if len(faces) == 0: # No face detected
                    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.02, minNeighbors=9) # Try again with different parameters
                if len(faces) == 0: # Still no face detected
                    continue
                if len(faces) > 1: # More than one face detected
                    # Choose the most prominent face
                    face = max(faces, key=lambda x: x[2] * x[3])
                    faces = [face]

                # Crop the face
                for (x, y, w, h) in faces:
                    # Extract face from the frame
                    face = frame[y:y+h, x:x+w]

                    # Save the frame to the disk
                    cv2.imwrite(os.path.join(video_path, f"{current_datetime_str}_{frames_extracted}.jpg"), face)
        
                    # Resize face
                    face = cv2.resize(face, IMG_SIZE)
                    img = Image.fromarray(face)
                    img = val_transform(img).unsqueeze(0)
                    img = img.to(device)

                    # Get prediction from model
                    output = model(img)

                    pred = torch.argmax(output, -1).detach()
                    emotion = merged_emotion_mapping[pred.item()] if use_positive_negative_labels else general_emotion_mapping[pred.item()]

                    # Return frame duration (float in seconds) and output (numpy array)
                    video_output.append({
                        'frame_duration': frame_duration, 
                        'emotion_label': pred.item(),
                        'emotion_string': emotion,
                        'logits': torch.softmax(output, -1).cpu().detach().numpy()
                        })
        
                frames_extracted += 1

        cap.release()
        cv2.destroyAllWindows()

    for emotion in video_output:
        print(f"Emotion detected at {emotion['frame_duration']:.2f}s: {emotion['emotion_string']}")

    return video_output

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
    hidden_size = HIDDEN_SIZE if configurations is None else configurations["hidden_size"]
    num_classes = NUM_CLASSES if configurations is None else configurations["num_classes"]
    dropout_p = DROPOUT_P if configurations is None else configurations["dropout_p"]
    
    model = select_model(type, hidden_size, num_classes, dropout_p).to(device)

    return model, num_classes