import torch
from config import RANDOM_SEED, USE_WANDB, VAL_SIZE, LIMIT, MODEL_NAME, BATCH_SIZE, LR, N_EPOCHS, METADATA_CSV, REG, NUM_CLASSES, DROPOUT_P, RESUME_TRAINING, PATH_TO_SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH, BALANCE_DATASET
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
                                   val_size=VAL_SIZE,
                                   seed=RANDOM_SEED,
                                   limit=LIMIT,
                                   balance_dataset=BALANCE_DATASET)
    
    train_loader, val_loader = fer_dataloader.get_train_val_dataloader()
    
    if MODEL_NAME == 'resnet18' or MODEL_NAME == 'resnet34' or MODEL_NAME == 'resnet50' or MODEL_NAME == 'resnet101':
        model = VideoResNetX(MODEL_NAME, NUM_CLASSES, DROPOUT_P).to(device)
    elif MODEL_NAME == 'dense121':
        model = VideoDenseNet121(NUM_CLASSES, DROPOUT_P).to(device)
    elif MODEL_NAME == 'custom_cnn':
        model = VideoCustomCNN(NUM_CLASSES, DROPOUT_P).to(device)
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
        "architecture": "VideoNet",
        "scope": "VideoNet",
        "model_name": MODEL_NAME,
        "learning_rate": LR,
        "epochs": N_EPOCHS,
        "reg": REG,
        "batch_size": BATCH_SIZE,
        "num_classes": NUM_CLASSES,
        "dataset": "FER2013",
        "optimizer": "AdamW",
        "resumed": RESUME_TRAINING,
        "use_wandb": USE_WANDB,
        "balance_dataset": "",
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