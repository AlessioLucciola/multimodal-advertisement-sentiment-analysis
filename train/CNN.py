import torch
from config import BATCH_SIZE, LR, N_EPOCHS, RANDOM_SEED, RAVDESS_CSV, RAVDESS_FILES, REG, RESUME_TRAINING, USE_WANDB
from dataloaders.RAVDESS_dataloader import get_train_val_dataloaders
from datasets.RAVDESS_dataset import RAVDESSCustomDataset
from models.CNN import CNN
from train.loops.train_loop import train_eval_loop
from utils.utils import set_seed, select_device

def main():
    set_seed(RANDOM_SEED)
    device = select_device()
    ravdess_dataset = RAVDESSCustomDataset(csv_file=RAVDESS_CSV, files_dir=RAVDESS_FILES)
    train_loader, val_loader, _ = get_train_val_dataloaders(ravdess_dataset, batch_size=BATCH_SIZE)
    model = CNN(num_emotions=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=REG)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-4, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    config = {
        "architecture": "CNN",
        "scope": "voice_emotion_recognition",
        "learning_rate": LR,
        "epochs": N_EPOCHS,
        'reg': REG,
        'batch_size': BATCH_SIZE,
        "hidden_size": "",
        "num_classes": "",
        "dataset": "",
        "optimizer": "",
        "dropout_p": "",
        "resumed": RESUME_TRAINING,
        "use_wandb": USE_WANDB,
    }

    train_eval_loop(device=device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model=model,
                    config=config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    resume=False)

if __name__ == "__main__":
    main()