from utils.utils import save_results, set_seed, select_device, upload_scaler
from config import AUDIO_FILES_DIR, BALANCE_DATASET, LIMIT, METADATA_ALL_CSV, METADATA_RAVDESS_CSV, PRELOAD_AUDIO_FILES, RAVDESS_FILES_DIR, REG, SAVE_RESULTS, RANDOM_SEED, PATH_TO_SAVE_RESULTS, NUM_CLASSES, BATCH_SIZE, SCALE_AUDIO_FILES, USE_RAVDESS_ONLY
from torchmetrics import Accuracy, Recall, Precision, F1Score, AUROC
from dataloaders.voice_custom_dataloader import RAVDESSDataLoader
from models.AudioNetCT import AudioNet_CNN_Transformers as AudioNetCT
from tqdm import tqdm
import torch
import os
import json

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
            if type == "AudioNet":
                test_data, test_labels = tr_batch['audio'], tr_batch['emotion'] # data = audio, labels = emotions
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

def get_model_and_dataloader(model_path, device):
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

    type = model_path.split('_')[0]
    model = None
    dataloader = None
    scaler = None
    if type == "AudioNet":
        num_classes = NUM_CLASSES if configurations is None else configurations["num_classes"]
        model = AudioNetCT(
            num_classes=num_classes).to(device)
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
    else:
        raise ValueError(f"Unknown architecture {type}")

    return model, dataloader, scaler, num_classes

def load_test_model(model, model_path, epoch, device):
    state_dict = torch.load(
        f"{PATH_TO_SAVE_RESULTS}/{model_path}/models/mi_project_{epoch}.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main(model_path, epoch):
    set_seed(RANDOM_SEED)
    device = select_device()
    model, dataloader, scaler, num_classes = get_model_and_dataloader(model_path, device)
    model = load_test_model(model, model_path, epoch, device)
    test_loader = dataloader.get_test_dataloader(scaler=scaler)
    criterion = torch.nn.CrossEntropyLoss()
    test_loop(model, test_loader, device, model_path, criterion, num_classes)


if __name__ == "__main__":
    # Name of the sub-folder into "results" folder in which to find the model to test (e.g. "resnet34_2023-12-10_12-29-49")
    model_path = "AudioNet_2024-03-27_10-23-10"
    # Specify the epoch number (e.g. 2) or "best" to get best model
    epoch = "1"

    main(model_path, epoch)