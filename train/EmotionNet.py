import torch
from config import RANDOM_SEED, USE_WANDB, VAL_SIZE, LIMIT, MODEL_NAME, BATCH_SIZE, LR, N_EPOCHS, METADATA_CSV, REG, FER_NUM_CLASSES, DROPOUT_P, RESUME_TRAINING, PATH_TO_SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH, BALANCE_DATASET, DATASET_NAME
from dataloaders.GREX_dataloader import GREXDataLoader
from train.loops.train_loop import train_eval_loop
from utils.utils import set_seed, select_device
from models.EmotionNet import EmotionNet
from models.PreProcessedEmotionNet import PreProcessedEmotionNet


def main():
    set_seed(RANDOM_SEED)
    device = select_device()
    loader = GREXDataLoader(batch_size=BATCH_SIZE)

    train_loader = loader.get_train_dataloader()
    val_loader = loader.get_val_dataloader()

    if MODEL_NAME == "PreProcessedEmotionNet":
        print("Using PreProcessedEmotionNet")
        model = PreProcessedEmotionNet(
            input_size=17,
            hidden_size=256,
            num_classes=5).to(device)
    elif MODEL_NAME == "EmotionNet":
        print("Using EmotionNet")
        model = EmotionNet(num_classes=5, dropout=DROPOUT_P).to(device)
    else:
        raise ValueError("Invalid model name")

    # if RESUME_TRAINING:
    #     model.load_state_dict(torch.load(
    #         f"{PATH_TO_SAVE_RESULTS}/{PATH_MODEL_TO_RESUME}/models/mi_project_{RESUME_EPOCH}.pt"))
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR,
                                 weight_decay=REG)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=LR,
        epochs=N_EPOCHS,
        steps_per_epoch=len(train_loader))

    criterion = torch.nn.CrossEntropyLoss()

    config = {
        "architecture": "EmotionNet",
        "scope": "EmotionNet",
        "learning_rate": LR,
        "epochs": N_EPOCHS,
        "reg": REG,
        "batch_size": BATCH_SIZE,
        "num_classes": FER_NUM_CLASSES,
        "dataset": DATASET_NAME,
        "optimizer": "AdamW",
        "resumed": RESUME_TRAINING,
        "use_wandb": USE_WANDB,
        "balance_dataset": BALANCE_DATASET,
        "limit": LIMIT,
        "dropout_p": DROPOUT_P
    }

    train_eval_loop(device=device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model=model,
                    config=config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    scaler=None,
                    resume=RESUME_TRAINING)


if __name__ == "__main__":
    main()
