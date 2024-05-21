from utils.utils import save_results, set_seed, select_device
from config import *
from torchmetrics import Accuracy, Recall
from tqdm import tqdm
import torch
import torch.nn as nn
import os
import json
from dataloaders.DEAP_dataloader import DEAPDataLoader
from models.EmotionNetDEAP import EmotionNet
from fusion.ppg_processing import main as ppg_main


def test_loop(model, device, model_path, num_classes):
    test_loader = DEAPDataLoader(batch_size=32).get_test_dataloader()
    criterion = nn.CrossEntropyLoss()

    model.eval()
    losses = []
    accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    recall_metric = Recall(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)
    
    pbar = tqdm(test_loader, desc="Testing")
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for batch in test_loader:
            pbar.update(1)
            src, target = batch["ppg"], batch["valence"]

            src = src.float().to(device)
            target = target.long().to(device)

            output = model(src)

            loss = criterion(output, target.long())
            losses.append(loss.item())
            
            preds = output.argmax(1)
            accuracy_metric.update(preds, target)
            recall_metric.update(preds, target)
            pbar.set_postfix_str(f"Test | Loss: {torch.tensor(losses).mean():.2f} | Acc: {(accuracy_metric.compute() * 100):.2f} | Rec: {(recall_metric.compute() * 100):.2f}")

        test_results = {
            "test_loss": torch.tensor(losses).mean().item(),
            "test_accuracy": accuracy_metric.compute().item(),
            "test_recall": recall_metric.compute().item(),
        }

        print(
            f"Test | Loss: {torch.tensor(losses).mean():.4f} | Accuracy: {(accuracy_metric.compute() * 100):.4f} | Recall: {(recall_metric.compute() * 100):.4f}"
        )

        if SAVE_RESULTS:
            save_results(model_path, test_results, test=True)



def get_model_and_dataloader(model_path, device):
    # Load configuration
    conf_path = PATH_TO_SAVE_RESULTS + f"/{model_path}/configurations.json"
    configurations = None
    if os.path.exists(conf_path):
        print(
            "--Model-- Old configurations found. Using those configurations for the test."
        )
        with open(conf_path, "r") as json_file:
            configurations = json.load(json_file)
    else:
        print(
            "--Model-- Old configurations NOT found. Using configurations in the config for test."
        )

    model = EmotionNet(dropout_p=configurations["dropout_p"] if configurations else DROPOUT_P).to(device)
    return model


def load_test_model(model, model_path, epoch, device):
    state_dict = torch.load(
        f"{PATH_TO_SAVE_RESULTS}/{model_path}/models/mi_project_{epoch}.pt",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def test_from_video(model_path, epoch):
    video_path = "/Users/dov/Desktop/wip-projects/multimodal-interaction-project/packages/rppg_toolbox/data/InferenceVideos/RawData/video1/my_video.mp4"
    return ppg_main(model_path=model_path, epoch=epoch, video_frames=video_path)

def main(model_path, epoch):
    set_seed(RANDOM_SEED)
    device = select_device()
    model = get_model_and_dataloader(model_path, device)
    model = load_test_model(model, model_path, epoch, device)
    test_loop(model, device, model_path, EMOTION_NUM_CLASSES)

if __name__ == "__main__":
    # Name of the sub-folder into "results" folder in which to find the model to test (e.g. "resnet34_2023-12-10_12-29-49")
    model_path = PATH_MODEL_TO_TEST[0] 
    epoch = TEST_EPOCH
    main(model_path, epoch)
