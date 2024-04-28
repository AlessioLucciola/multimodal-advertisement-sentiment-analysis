from utils.utils import save_results, set_seed, select_device
from config import *
from torchmetrics import Accuracy, Recall
from tqdm import tqdm
import torch
import os
import json
from dataloaders.CEAP_dataloader import CEAPDataLoader
from models.EmotionNetCEAP import EmotionNet, Encoder, Decoder


def test_loop(model, test_loader, device, model_path, criterion, num_classes):
    model.eval()
    losses = []
    accuracy_metric = Accuracy(
        task="multiclass", num_classes=num_classes).to(device)
    recall_metric = Recall(
        task="multiclass", num_classes=num_classes, average='macro').to(device)

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            src, target = batch["ppg"], batch["valence"]

            src = src.float().to(device)
            target = target.float().to(device)

            src = src.permute(1, 0, 2)
            target = target.permute(1, 0)

            output = model(src, target, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = target[1:].reshape(-1)
            loss = criterion(output, trg)
            losses.append(loss.item())

            preds = output.argmax(dim=1)
            accuracy_metric.update(preds, trg)
            recall_metric.update(preds, trg)

        test_results = {
            'test_loss': torch.tensor(losses).mean().item(),
            'test_accuracy': accuracy_metric.compute().item(),
            'test_recall': recall_metric.compute().item(),
        }

        print(f"Test | Loss: {torch.tensor(losses).mean():.4f} | Accuracy: {(accuracy_metric.compute() * 100):.4f} | Recall: {(recall_metric.compute() * 100):.4f}")

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

    input_dim = LENGTH // WAVELET_STEP if WT else 1
    output_dim = 3
    encoder_embedding_dim = LENGTH // WAVELET_STEP if WT else 1
    decoder_embedding_dim = LENGTH // WAVELET_STEP if WT else 1
    hidden_dim = LSTM_HIDDEN if configurations is None else configurations["lstm_config"]["num_hidden"]
    n_layers = LSTM_LAYERS if configurations is None else configurations["lstm_config"]["num_layers"]
    encoder_dropout = DROPOUT_P 
    decoder_dropout = DROPOUT_P
    num_classes = EMOTION_NUM_CLASSES

    encoder = Encoder(
        input_dim,
        encoder_embedding_dim,
        hidden_dim,
        n_layers,
        encoder_dropout,
    )

    decoder = Decoder(
        output_dim,
        decoder_embedding_dim,
        hidden_dim,
        n_layers,
        decoder_dropout,
    )

    model = EmotionNet(encoder, decoder).to(device)

    dataloader = CEAPDataLoader(batch_size=32)

    return model, dataloader, num_classes


def load_test_model(model, model_path, epoch, device):
    state_dict = torch.load(
        f"{PATH_TO_SAVE_RESULTS}/{model_path}/models/mi_project_{epoch}.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main(model_path, epoch):
    set_seed(RANDOM_SEED)
    device = select_device()
    model, dataloader, num_classes = get_model_and_dataloader(model_path, device)
    model = load_test_model(model, model_path, epoch, device)

    test_loader = dataloader.get_test_dataloader()
    criterion = torch.nn.CrossEntropyLoss()
    test_loop(model, test_loader, device, model_path, criterion, num_classes)


if __name__ == "__main__":
    # Name of the sub-folder into "results" folder in which to find the model to test (e.g. "resnet34_2023-12-10_12-29-49")
    model_path = "EmotionNet - LSTM Seq2Seq_2024-04-27_17-09-45"
    # Specify the epoch number (e.g. 2) or "best" to get best model
    epoch = "246"

    main(model_path, epoch)
