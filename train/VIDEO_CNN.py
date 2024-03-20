import torch
from config import RANDOM_SEED, USE_WANDB
from video_config import MODEL_NAME, BATCH_SIZE, LR, N_EPOCHS, RANDOM_SEED, METADATA_CSV, REG, NUM_CLASSES
from dataloaders.FER_dataloader import FERDataloader
from models.VIDEO_CNN import get_model
from train.loops.train_loop import train_eval_loop
from utils.utils import set_seed, select_device

def main():
    set_seed(RANDOM_SEED)
    device = select_device()
    train_loader, val_loader, test_loader = FERDataloader(data_dir=METADATA_CSV,
                                           batch_size=BATCH_SIZE)
    
    model = get_model(NUM_CLASSES, device, model_name= MODEL_NAME)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=REG)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=LR,
                                                    epochs=N_EPOCHS,
                                                    steps_per_epoch=len(train_loader))                                   
    criterion = torch.nn.CrossEntropyLoss()

    # TO DO: Complete the configuration of the model
    config = {
        "architecture": "VIDEO_CNN",
        "scope": "video_emotion_recognition",
        "learning_rate": LR,
        "epochs": N_EPOCHS,
        'reg': REG,
        'batch_size': BATCH_SIZE,
        "hidden_size": "",
        "num_classes": "",
        "dataset": "",
        "optimizer": "",
        "dropout_p": "",
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