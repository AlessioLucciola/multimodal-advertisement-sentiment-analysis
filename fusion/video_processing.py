from config import AUDIO_SAMPLE_RATE, AUDIO_OFFSET, AUDIO_DURATION, DROPOUT_P, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, NUM_MFCC, FRAME_LENGTH, HOP_LENGTH, PATH_TO_SAVE_RESULTS, RAVDESS_NUM_CLASSES, RANDOM_SEED, METADATA_CSV, FER_NUM_CLASSES, BATCH_SIZE, LIMIT, BALANCE_DATASET, AUGMENT_DATASET, USE_DEFAULT_SPLIT, MODEL_NAME
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
from shared.constants import FER_emotion_mapping
from models.VideoDenseNet121 import VideoDenseNet121
from models.VideoResnetX import VideoResNetX
from models.VideoCustomCNN import VideoCustomCNN

def main(model_path, video_file_path, epoch):
    set_seed(RANDOM_SEED)
    device = select_device()
    type = model_path.split('_')[0]
    model, scaler, _ = get_model_and_dataloader(model_path, device, type)
    model = load_test_model(model, model_path, epoch, device)
    
    val_transform = transforms.Compose([
        transforms.ToTensor()])
    
    # Load the video file
    video = cv2.VideoCapture(video_file_path)

    count = 0

    # Loop through each frame in the video
    while True:
        ret, frame = video.read()

        if not ret:
            break
        # For every 1000 frames, run the model
        if count % 10 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier('./models/haarcascade/haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
                resize_frame = cv2.resize(gray[y:y + h, x:x + w], (48, 48)) 
                X = resize_frame/256
                X = Image.fromarray((X))
                X = val_transform(X).unsqueeze(0)
                with torch.no_grad():
                    model.eval()
                    log_ps = model.cpu()(X)
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    pred = FER_emotion_mapping[int(top_class.numpy())]
                cv2.putText(frame, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
            
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        count = count + 1

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
    
    num_classes = FER_NUM_CLASSES if configurations is None else configurations["num_classes"]
    dropout_p = DROPOUT_P if configurations is None else configurations["dropout_p"]
    # Set the model
    if MODEL_NAME == 'resnet18' or MODEL_NAME == 'resnet34' or MODEL_NAME == 'resnet50' or MODEL_NAME == 'resnet101':
        model = VideoResNetX(MODEL_NAME, FER_NUM_CLASSES, DROPOUT_P).to(device)
    elif MODEL_NAME == 'dense121':
        model = VideoDenseNet121(FER_NUM_CLASSES, DROPOUT_P).to(device)
    elif MODEL_NAME == 'custom_cnn':
        model = VideoCustomCNN(FER_NUM_CLASSES, DROPOUT_P).to(device)
    else:
        raise ValueError('Invalid Model Name: Options [resnet18, resnet34, resnet50, resnet101, dense121, custom_cnn]')
    
    scaler = None

    return model, scaler, num_classes

if __name__ == "__main__":
    epoch = 20
    model_path = os.path.join("VideoNet_resnet34_2024-04-05_16-05-38")
    video_file_path = os.path.join("data", "VIDEO", "test_video.mp4")
    main(model_path=model_path, video_file_path=video_file_path, epoch=epoch)