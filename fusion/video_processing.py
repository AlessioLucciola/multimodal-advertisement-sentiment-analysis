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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main(model_path, video_frames, epoch, live_demo):
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
    video_frames = get_frames_duration(video_frames, live_demo) 
    for duration, video_frame in video_frames:
        # Trasform _io.BytesIO object to PIL Image object and then to tensor object to pass to the model
        video_frame = Image.open(video_frame).convert('RGB')
        # Transform to tensor
        video_frame = video_frame.resize((224, 224))
        video_frame = np.array(video_frame)
        video_frame = torch.tensor(video_frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
        output = model(video_frame)
        # Save the video frame to the disk
        image_array = np.array(video_frame)
        height, width, channels = image_array.shape
        pixels = image_array.reshape((height * width, channels))
        pixels = pixels.astype(np.uint8)
        image = Image.fromarray(pixels.reshape((height, width, channels)))
        image.save(os.path.join(video_path, f"{current_datetime_str}_{duration}.jpg"))
        # In the real code, for each frame you should return the logits of each emotion
        video_output.append({'frame_duration': duration, 'output': output})

    return video_output

def get_frames_duration(video_frames, live_demo):
    if live_demo:
        start_time = datetime.timestamp(video_frames[0][1])
        frame_duration = [(datetime.timestamp(frame[1]) - start_time, frame[0]) for frame in video_frames]
    else:
        frame_duration = [(frame[1], frame[0]) for frame in video_frames]
    return frame_duration

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