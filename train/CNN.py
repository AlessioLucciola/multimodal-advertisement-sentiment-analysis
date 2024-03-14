import torch
from config import BATCH_SIZE, LR, N_EPOCHS, RANDOM_SEED, RAVDESS_CSV, RAVDESS_FILES, REG
from dataloaders.RAVDESS_dataloader import get_train_val_dataloaders
from datasets.RAVDESS_dataset import RAVDESSCustomDataset
from models.CNN import CNN
from utils.utils import set_seed

def main():
    set_seed(RANDOM_SEED)
    ravdess_dataset = RAVDESSCustomDataset(csv_file=RAVDESS_CSV, files_dir=RAVDESS_FILES)
    train_loader, val_loader, _ = get_train_val_dataloaders(ravdess_dataset, batch_size=BATCH_SIZE)
    model = CNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=REG)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-4, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

if __name__ == "__main__":
    main()