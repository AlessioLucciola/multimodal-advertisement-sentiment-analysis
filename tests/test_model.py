from utils.utils import save_results, set_seed, select_device, upload_scaler
from config import *
from torchmetrics import Accuracy, Recall, Precision, F1Score, AUROC
from dataloaders.voice_custom_dataloader import RAVDESSDataLoader
from models.AudioNetCT import AudioNet_CNN_Transformers as AudioNetCT
from models.AudioNetCL import AudioNet_CNN_LSTM as AudioNetCL
from tqdm import tqdm
import torch
import os
import json
import torchvision.transforms as transforms
import cv2
from PIL import Image
from shared.constants import FER_emotion_mapping
from dataloaders.FER_dataloader import FERDataloader
from models.VideoDenseNet121 import VideoDenseNet121
from models.VideoResnetX import VideoResNetX
from models.VideoCustomCNN import VideoCustomCNN

def test_loop(test_model, test_loader, device, model_path, criterion, num_classes):
    test_model.eval()
    test_loss_iter = 0
    accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    recall_metric = Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)
    precision_metric = Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
    f1_metric = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)
    auroc_metric = AUROC(task="multiclass", num_classes=num_classes).to(device)

    with torch.no_grad():
        epoch_test_preds = torch.tensor([]).to(device)
        epoch_test_labels = torch.tensor([], dtype=torch.long).to(device)
        epoch_test_probs = torch.tensor([]).to(device)
        for _, tr_batch in enumerate(tqdm(test_loader, desc="Testing model..", leave=False)):
            type = model_path.split('_')[0]
            if type == "AudioNetCT" or type == "AudioNetCL":
                test_data, test_labels = tr_batch['audio'], tr_batch['emotion'] # data = audio, labels = emotions
            if type == "VideoNet":
                test_data, test_labels = tr_batch[0], tr_batch[1] # data = pixel, labels = emotions
            test_data = test_data.to(device)
            test_labels = test_labels.to(device)

            test_outputs = test_model(test_data)
            test_preds = torch.argmax(test_outputs, -1).detach()
            test_probs = test_outputs.softmax(dim=1)
            epoch_test_preds = torch.cat((epoch_test_preds, test_preds), 0)
            epoch_test_labels = torch.cat((epoch_test_labels, test_labels), 0)
            epoch_test_probs = torch.cat((epoch_test_probs, test_probs), 0)

            # Multiclassification loss considering all classes
            test_epoch_loss = criterion(test_outputs, test_labels)
            test_loss_iter += test_epoch_loss.item()

        test_loss = test_loss_iter / (len(test_loader) * test_loader.batch_size)
        test_accuracy = accuracy_metric(epoch_test_preds, epoch_test_labels) * 100
        test_recall = recall_metric(epoch_test_preds, epoch_test_labels) * 100
        test_precision = precision_metric(epoch_test_preds, epoch_test_labels) * 100
        test_f1 = f1_metric(epoch_test_preds, epoch_test_labels) * 100
        test_auroc = auroc_metric(epoch_test_probs, epoch_test_labels) * 100

        print('Test -> Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%, Precision: {:.4f}%, F1: {:.4f}%, AUROC: {:.4f}%'.format(
            test_loss, test_accuracy, test_recall, test_precision, test_f1, test_auroc))

        test_results = {
            'test_loss': test_loss,
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
    if type == "AudioNetCT":
        num_classes = RAVDESS_NUM_CLASSES if configurations is None else configurations["num_classes"]
        num_mfcc = NUM_MFCC if configurations is None else configurations["num_mfcc"]
        dropout_p = DROPOUT_P if configurations is None else configurations["dropout_p"]
        model = AudioNetCT(
            num_classes=num_classes, num_mfcc=num_mfcc, dropout_p=dropout_p).to(device)
        dataloader = RAVDESSDataLoader(csv_file=METADATA_RAVDESS_CSV if USE_RAVDESS_ONLY else METADATA_ALL_CSV,
                                        audio_files_dir=RAVDESS_FILES_DIR if USE_RAVDESS_ONLY else AUDIO_FILES_DIR,
                                        batch_size=BATCH_SIZE,
                                        seed=RANDOM_SEED,
                                        limit=LIMIT,
                                        balance_dataset=BALANCE_DATASET,
                                        preload_audio_files=PRELOAD_AUDIO_FILES,
                                        scale_audio_files=SCALE_AUDIO_FILES
                                        )
        scaler = upload_scaler(model_path)
    elif type == "AudioNetCL":
        num_classes = RAVDESS_NUM_CLASSES if configurations is None else configurations["num_classes"]
        num_mfcc = NUM_MFCC if configurations is None else configurations["num_mfcc"]
        lstm_hidden_size = LSTM_HIDDEN_SIZE if configurations is None else configurations["lstm_hidden_size"]
        lstm_num_layers = LSTM_NUM_LAYERS if configurations is None else configurations["lstm_num_layers"]
        dropout_p = DROPOUT_P if configurations is None else configurations["dropout_p"]
        model = AudioNetCL(
            num_classes=num_classes, num_mfcc=num_mfcc, num_layers=lstm_num_layers, hidden_size=lstm_hidden_size, dropout_p=dropout_p).to(device)
        dataloader = RAVDESSDataLoader(csv_file=METADATA_RAVDESS_CSV if USE_RAVDESS_ONLY else METADATA_ALL_CSV,
                                        audio_files_dir=RAVDESS_FILES_DIR if USE_RAVDESS_ONLY else AUDIO_FILES_DIR,
                                        batch_size=BATCH_SIZE,
                                        seed=RANDOM_SEED,
                                        limit=LIMIT,
                                        balance_dataset=BALANCE_DATASET,
                                        preload_audio_files=PRELOAD_AUDIO_FILES,
                                        scale_audio_files=SCALE_AUDIO_FILES
                                        )
        scaler = upload_scaler(model_path)
    elif type == "VideoNet":
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
        
        dataloader = FERDataloader(csv_file=METADATA_CSV,
                                   batch_size=BATCH_SIZE,
                                   seed=RANDOM_SEED,
                                   limit=LIMIT,
                                   balance_dataset=BALANCE_DATASET,
                                   use_default_split=USE_DEFAULT_SPLIT)
        scaler = None
    else:
        raise ValueError(f"Unknown architecture {type}")

    return model, dataloader, scaler, num_classes

def load_test_model(model, model_path, epoch, device):
    state_dict = torch.load(
        f"{PATH_TO_SAVE_RESULTS}/{model_path}/models/mi_project_{epoch}.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def video_live_test(model):
    val_transform = transforms.Compose([
        transforms.ToTensor()])

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
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

    cap.release()
    cv2.destroyAllWindows()


def main(model_path, epoch):
    set_seed(RANDOM_SEED)
    device = select_device()
    type = model_path.split('_')[0]
    model, dataloader, scaler, num_classes = get_model_and_dataloader(model_path, device, type)
    model = load_test_model(model, model_path, epoch, device)

    if LIVE_TEST:
        if type == "VideoNet":
            video_live_test(model)
    else:
        test_loader = dataloader.get_test_dataloader(scaler=scaler) if type == "AudioNetCT" or type == "AudioNetCL" else dataloader.get_test_dataloader()
        criterion = torch.nn.CrossEntropyLoss()
        test_loop(model, test_loader, device, model_path, criterion, num_classes)

if __name__ == "__main__":
    # Name of the sub-folder into "results" folder in which to find the model to test (e.g. "resnet34_2023-12-10_12-29-49")
    model_path = "AudioNetCT_2024-04-08_09-33-56"
    # Specify the epoch number (e.g. 2) or "best" to get best model
    epoch = "484"

    main(model_path, epoch)