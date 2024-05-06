from utils.utils import save_results, set_seed, select_device, upload_scaler
from config import *
from torchmetrics import Accuracy, Recall, Precision, F1Score, AUROC
from dataloaders.voice_custom_dataloader import RAVDESSDataLoader
from dataloaders.ravdess_custom_dataloader import ravdess_custom_dataloader
from models.AudioNetCT import AudioNet_CNN_Transformers as AudioNetCT
from models.AudioNetCL import AudioNet_CNN_LSTM as AudioNetCL
from tqdm import tqdm
import torch
import os
import json
import torchvision.transforms as transforms
import cv2
from PIL import Image
from shared.constants import merged_emotion_mapping, general_emotion_mapping
from utils.video_utils import select_model

def test_loop(test_model, test_loader, device, model_path, criterion, num_classes):
    test_model.eval()
    accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    recall_metric = Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)
    precision_metric = Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
    f1_metric = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)
    auroc_metric = AUROC(task="multiclass", num_classes=num_classes).to(device)

    with torch.no_grad():
        epoch_test_preds = torch.tensor([]).to(device)
        epoch_test_labels = torch.tensor([], dtype=torch.long).to(device)
        epoch_test_probs = torch.tensor([]).to(device)
        epoch_test_loss = 0
        for _, tr_batch in enumerate(tqdm(test_loader, desc="Testing model..", leave=False)):
            type = model_path.split('_')[0]
            if type == "AudioNetCT" or type == "AudioNetCL":
                test_data, test_labels = tr_batch['audio'], tr_batch['emotion'] # data = audio, labels = emotions
            if type == "VideoNet":
                test_data, test_labels = tr_batch['frame'], tr_batch['emotion'] # data = pixel, labels = emotions
            test_data = test_data.to(device)
            test_labels = test_labels.to(device)

            test_outputs = test_model(test_data)
            test_preds = torch.argmax(test_outputs, -1).detach()
            test_probs = test_outputs.softmax(dim=1)
            epoch_test_preds = torch.cat((epoch_test_preds, test_preds), 0)
            epoch_test_labels = torch.cat((epoch_test_labels, test_labels), 0)
            epoch_test_probs = torch.cat((epoch_test_probs, test_probs), 0)

            # Multiclassification loss considering all classes
            test_loss = criterion(test_outputs, test_labels)
            epoch_test_loss += test_loss.item()


        final_test_loss = epoch_test_loss / len(test_loader)
        test_accuracy = accuracy_metric(epoch_test_preds, epoch_test_labels) * 100
        test_recall = recall_metric(epoch_test_preds, epoch_test_labels) * 100
        test_precision = precision_metric(epoch_test_preds, epoch_test_labels) * 100
        test_f1 = f1_metric(epoch_test_preds, epoch_test_labels) * 100
        test_auroc = auroc_metric(epoch_test_probs, epoch_test_labels) * 100

        print('Test -> Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%, Precision: {:.4f}%, F1: {:.4f}%, AUROC: {:.4f}%'.format(
            final_test_loss, test_accuracy, test_recall, test_precision, test_f1, test_auroc))

        test_results = {
            'test_loss': final_test_loss,
            'test_accuracy': test_accuracy.item(),
            'test_recall': test_recall.item(),
            'test_precision': test_precision.item(),
            'test_f1': test_f1.item(),
            'test_auroc': test_auroc.item()
        }
        if SAVE_RESULTS:
            save_results(model_path, test_results, test=True)

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

    model = None
    dataloader = None
    scaler = None
    num_classes = None
    use_positive_negative_labels = None
    if type == "AudioNetCT":
        num_classes = NUM_CLASSES if configurations is None else configurations["num_classes"]
        num_mfcc = NUM_MFCC if configurations is None else configurations["num_mfcc"]
        dropout_p = DROPOUT_P if configurations is None else configurations["dropout_p"]
        model = AudioNetCT(
            num_classes=num_classes, num_mfcc=num_mfcc, dropout_p=dropout_p).to(device)
        dataloader = RAVDESSDataLoader(csv_file=AUDIO_METADATA_RAVDESS_CSV if USE_RAVDESS_ONLY else AUDIO_METADATA_ALL_CSV,
                                        audio_files_dir=AUDIO_RAVDESS_FILES_DIR if USE_RAVDESS_ONLY else AUDIO_FILES_DIR,
                                        batch_size=BATCH_SIZE,
                                        seed=RANDOM_SEED,
                                        limit=LIMIT,
                                        balance_dataset=BALANCE_DATASET,
                                        preload_audio_files=PRELOAD_AUDIO_FILES,
                                        scale_audio_files=SCALE_AUDIO_FILES,
                                        use_positive_negative_labels=USE_POSITIVE_NEGATIVE_LABELS
                                        )
        scaler = upload_scaler(model_path)
    elif type == "AudioNetCL":
        num_classes = NUM_CLASSES if configurations is None else configurations["num_classes"]
        num_mfcc = NUM_MFCC if configurations is None else configurations["num_mfcc"]
        lstm_hidden_size = LSTM_HIDDEN_SIZE if configurations is None else configurations["lstm_hidden_size"]
        lstm_num_layers = LSTM_NUM_LAYERS if configurations is None else configurations["lstm_num_layers"]
        dropout_p = DROPOUT_P if configurations is None else configurations["dropout_p"]
        model = AudioNetCL(
            num_classes=num_classes, num_mfcc=num_mfcc, num_layers=lstm_num_layers, hidden_size=lstm_hidden_size, dropout_p=dropout_p).to(device)
        dataloader = RAVDESSDataLoader(csv_file=AUDIO_METADATA_RAVDESS_CSV if USE_RAVDESS_ONLY else AUDIO_METADATA_ALL_CSV,
                                        audio_files_dir=AUDIO_RAVDESS_FILES_DIR if USE_RAVDESS_ONLY else AUDIO_FILES_DIR,
                                        batch_size=BATCH_SIZE,
                                        seed=RANDOM_SEED,
                                        limit=LIMIT,
                                        balance_dataset=BALANCE_DATASET,
                                        preload_audio_files=PRELOAD_AUDIO_FILES,
                                        scale_audio_files=SCALE_AUDIO_FILES,
                                        use_positive_negative_labels=USE_POSITIVE_NEGATIVE_LABELS
                                        )
        scaler = upload_scaler(model_path)
    elif type == "VideoNet":
        hidden_size = HIDDEN_SIZE if configurations is None else configurations["hidden_size"]
        batch_size = BATCH_SIZE if configurations is None else configurations["batch_size"]
        num_classes = NUM_CLASSES if configurations is None else configurations["num_classes"]
        dropout_p = DROPOUT_P if configurations is None else configurations["dropout_p"]
        use_positive_negative_labels = USE_POSITIVE_NEGATIVE_LABELS if configurations is None else configurations["use_positive_negative_labels"]
        overlap_subjects_frames = OVERLAP_SUBJECTS_FRAMES if configurations is None else configurations["overlap_subjects_frames"]
        use_positive_negative_labels = USE_POSITIVE_NEGATIVE_LABELS if configurations is None else configurations["use_positive_negative_labels"]

        model = select_model(model_path.split('_')[1], hidden_size, num_classes, dropout_p).to(device)
        if not USE_VIDEO_FOR_TESTING:
            dataloader = ravdess_custom_dataloader(csv_original_files=VIDEO_METADATA_CSV,
                                   csv_frames_files=VIDEO_METADATA_FRAMES_CSV,
                                   batch_size=batch_size,
                                   frames_dir=FRAMES_FILES_DIR,
                                   seed=RANDOM_SEED,
                                   limit=LIMIT,
                                   overlap_subjects_frames=overlap_subjects_frames,
                                   use_positive_negative_labels=use_positive_negative_labels,
                                   preload_frames=PRELOAD_FRAMES,
                                   apply_transformations=APPLY_TRANSFORMATIONS,
                                   balance_dataset=BALANCE_DATASET,
                                   normalize=NORMALIZE,
                                   )
    else:
        raise ValueError(f"Unknown architecture {type}")
    
    return model, dataloader, scaler, num_classes, use_positive_negative_labels

def load_test_model(model, model_path, epoch, device):
    state_dict = torch.load(
        f"{PATH_TO_SAVE_RESULTS}/{model_path}/models/mi_project_{epoch}.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def video_test(model, cap, device, use_positive_negative_labels):
    model.eval()

    # Define the transformation
    val_transform = transforms.Compose([
        transforms.ToTensor()])
    
    # Load the face cascade
    face_cascade = cv2.CascadeClassifier('./models/haarcascade/haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.12, minNeighbors=9)
                    
        if len(faces) == 0: # No face detected
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.02, minNeighbors=9) # Try again with different parameters
            if len(faces) == 0: # Still no face detected
                continue
        if len(faces) > 1: # More than one face detected
            # Choose the most prominent face
            face = max(faces, key=lambda x: x[2] * x[3])
            faces = [face]

        for (x, y, w, h) in faces:
            # Extract face from the frame
            face = frame[y:y+h, x:x+w]
  
            # Resize face
            face = cv2.resize(face, IMG_SIZE)
            img = Image.fromarray(face)
            img = val_transform(img).unsqueeze(0)
            img = img.to(device)

            # Get prediction from model
            output = model(img)
            pred = torch.argmax(output, -1).detach()
            prob = output.softmax(dim=1).max().item()
            emotion = merged_emotion_mapping[pred.item()] if use_positive_negative_labels else general_emotion_mapping[pred.item()]

            # Display rectagnle with emotion and probability
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1) 
            cv2.putText(frame, f"{prob:.2f}", (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)   

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main(model_path):
    set_seed(RANDOM_SEED)
    device = select_device()
    type = model_path.split('_')[0]

    if type == "AudioNetCT" or type == "AudioNetCL":
        print("--Test-- Audio model test")
        epoch = AUDIO_MODEL_EPOCH
        model, dataloader, scaler, num_classes, _ = get_model_and_dataloader(model_path, device, type)
        model = load_test_model(model, model_path, epoch, device)
    elif type == "VideoNet":
        print("--Test-- Video model test")
        epoch = VIDEO_MODEL_EPOCH
        model, dataloader, scaler, num_classes, use_positive_negative_labels = get_model_and_dataloader(model_path, device, type)
        model = load_test_model(model, model_path, epoch, device)

        if USE_VIDEO_FOR_TESTING:
            if USE_LIVE_VIDEO_FOR_TESTING:
                print("--Test-- Live video test")
                cap = cv2.VideoCapture(0)
            else:
                print("--Test-- Offline video test")
                cap = cv2.VideoCapture(OFFLINE_VIDEO_FILE)
            video_test(model, cap, device, use_positive_negative_labels)
            return
    else:
        raise ValueError(f"Unknown architecture {type}")
            
    print("--Test-- Test dataset")
    test_loader = dataloader.get_test_dataloader(scaler=scaler) if type == "AudioNetCT" or type == "AudioNetCL" else dataloader.get_test_dataloader()
    criterion = torch.nn.CrossEntropyLoss()
    test_loop(model, test_loader, device, model_path, criterion, num_classes)

if __name__ == "__main__":
    model_paths = PATH_MODELS_TO_TEST
    for m in model_paths:
        main(m)