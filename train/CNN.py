import torch
from config import BATCH_SIZE, LR, N_EPOCHS, RANDOM_SEED, RAVDESS_CSV, RAVDESS_FILES, REG, RESUME_TRAINING, USE_WANDB, RAVDESS_NUM_CLASSES, LIMIT
from dataloaders.RAVDESS_dataloader import RAVDESSDataLoader
from models.CNN import CNN
from train.loops.train_loop import train_eval_loop
from utils.utils import set_seed, select_device

def main():
    set_seed(RANDOM_SEED)
    device = select_device()
    ravdess_dataloader = RAVDESSDataLoader(csv_file=RAVDESS_CSV, audio_files_dir=RAVDESS_FILES, batch_size=BATCH_SIZE, seed=RANDOM_SEED, limit=LIMIT)
    train_loader = ravdess_dataloader.get_train_dataloader()
    val_loader = ravdess_dataloader.get_val_dataloader()
    model = CNN(num_classes=RAVDESS_NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=REG)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-4, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    # TO DO: Complete the configuration of the model
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