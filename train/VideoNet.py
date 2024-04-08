import torch
from config import RANDOM_SEED, USE_WANDB, LIMIT, MODEL_NAME, BATCH_SIZE, LR, N_EPOCHS, METADATA_CSV, REG, FER_NUM_CLASSES, DROPOUT_P, RESUME_TRAINING, PATH_TO_SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH, BALANCE_DATASET, DATASET_NAME, USE_DEFAULT_SPLIT, APPLY_TRANSFORMATIONS, DF_SPLITTING
from dataloaders.FER_dataloader import FERDataloader
from train.loops.train_loop import train_eval_loop
from utils.utils import set_seed, select_device

from models.VideoDenseNet121 import VideoDenseNet121
from models.VideoResnetX import VideoResNetX
from models.VideoCustomCNN import VideoCustomCNN

def main():
    set_seed(RANDOM_SEED)
    device = select_device()
    fer_dataloader = FERDataloader(csv_file=METADATA_CSV,
                                   batch_size=BATCH_SIZE,
                                   seed=RANDOM_SEED,
                                   limit=LIMIT,
                                   apply_transformations=APPLY_TRANSFORMATIONS,
                                   balance_dataset=BALANCE_DATASET,
                                   use_default_split=USE_DEFAULT_SPLIT)
    
    train_loader = fer_dataloader.get_train_dataloader()
    val_loader = fer_dataloader.get_val_dataloader()
    
    if MODEL_NAME == 'resnet18' or MODEL_NAME == 'resnet34' or MODEL_NAME == 'resnet50' or MODEL_NAME == 'resnet101':
        model = VideoResNetX(MODEL_NAME, FER_NUM_CLASSES, DROPOUT_P).to(device)
    elif MODEL_NAME == 'dense121':
        model = VideoDenseNet121(FER_NUM_CLASSES, DROPOUT_P).to(device)
    elif MODEL_NAME == 'custom_cnn':
        model = VideoCustomCNN(FER_NUM_CLASSES, DROPOUT_P).to(device)
    else:
        raise ValueError('Invalid Model Name: Options [resnet18, resnet34, resnet50, resnet101, dense121, custom_cnn]')
    
    if RESUME_TRAINING:
        model.load_state_dict(torch.load(
            f"{PATH_TO_SAVE_RESULTS}/{PATH_MODEL_TO_RESUME}/models/mi_project_{RESUME_EPOCH}.pt"))
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
        "architecture": "VideoNet_" + MODEL_NAME,
        "scope": "VideoNet",
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
        "use_default_split": USE_DEFAULT_SPLIT,
        "df_splitting": None if USE_DEFAULT_SPLIT else DF_SPLITTING,
        "apply_transformations": APPLY_TRANSFORMATIONS,
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