from config import BATCH_SIZE, SAVE_MODELS, SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH, RANDOM_SEED, PATH_TO_SAVE_RESULTS
from utils.utils import save_results, save_model, save_configurations, save_scaler, select_device, set_seed

from torchmetrics import Accuracy, Recall
from models.EmotionNetCT import EmotionNet
from dataloaders.GREX_dataloader import GREXDataLoader
import torch


def test_loop(device,
              test_loader: torch.utils.data.DataLoader,
              num_classes,
              criterion):

    accuracy_metric = Accuracy(
        task="multiclass", num_classes=num_classes).to(device)
    recall_metric = Recall(
        task="multiclass", num_classes=num_classes,  average='macro').to(device)
    val_cumulative_loss = 0
    val_step = 0
    model.eval()
    with torch.no_grad():
        epoch_val_preds = torch.tensor([]).to(device)
        epoch_val_labels = torch.tensor([]).to(device)
        for _, val_batch in enumerate(test_loader):
            val_data, val_spatial, val_labels = val_batch["ppg"], val_batch[
                "ppg_spatial_features"], val_batch["valence"]

            val_data = val_data.float().to(device)
            val_spatial = val_spatial.float().to(device)
            val_labels = val_labels.to(device)

            val_outputs = model(val_data, val_spatial)

            val_preds = torch.argmax(val_outputs, -1).detach()

            epoch_val_preds = torch.cat((epoch_val_preds, val_preds), 0)
            epoch_val_labels = torch.cat((epoch_val_labels, val_labels), 0)

            val_epoch_loss = criterion(val_outputs, val_labels)

            val_cumulative_loss += val_epoch_loss
            val_step += 1

        val_accuracy = accuracy_metric(
            epoch_val_preds, epoch_val_labels) * 100
        val_recall = recall_metric(
            epoch_val_preds, epoch_val_labels) * 100

        val_cumulative_loss /= val_step

    print(
        f"Testing -> Loss: {val_cumulative_loss:.4f}), Accuracy: {val_accuracy: .4f} % , Recall: {val_recall: .4f} %")


if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    device = select_device()
    dataloader = GREXDataLoader(batch_size=64)
    model = EmotionNet(num_classes=3).to(device)
    test_loader = dataloader.get_test_dataloader()
    criterion = torch.nn.CrossEntropyLoss()
    # TODO: load model weight
    model_path = "EmotionNet - CNN + LSTM_2024-04-15_20-23-20"
    epoch = 20
    state_dict = torch.load(f"{PATH_TO_SAVE_RESULTS}/{model_path}/models/mi_project_{epoch}.pt", map_location=device)
    model.load_state_dict(state_dict)

    test_loop(device=device, test_loader=test_loader, criterion=criterion, num_classes=3)
