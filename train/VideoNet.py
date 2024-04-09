import torch
from config import RANDOM_SEED, USE_WANDB, LIMIT, MODEL_NAME, BATCH_SIZE, LR, N_EPOCHS, METADATA_CSV, REG, VIDEO_NUM_CLASSES, DROPOUT_P, RESUME_TRAINING, PATH_TO_SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH, BALANCE_DATASET, DATASET_NAME, USE_DEFAULT_SPLIT, APPLY_TRANSFORMATIONS, DF_SPLITTING, HIDDEN_SIZE, NORMALIZE
from dataloaders.video_custom_dataloader import video_custom_dataloader
from train.loops.train_loop import train_eval_loop
from utils.utils import set_seed, select_device
from utils.video_utils import select_model

def main():
    set_seed(RANDOM_SEED)
    device = select_device()
    fer_dataloader = video_custom_dataloader(csv_file=METADATA_CSV,
                                   batch_size=BATCH_SIZE,
                                   seed=RANDOM_SEED,
                                   limit=LIMIT,
                                   apply_transformations=APPLY_TRANSFORMATIONS,
                                   balance_dataset=BALANCE_DATASET,
                                   use_default_split=USE_DEFAULT_SPLIT,
                                   normalize=NORMALIZE,
                                   )
    
    train_loader = fer_dataloader.get_train_dataloader()
    val_loader = fer_dataloader.get_val_dataloader()
    
    model = select_model(MODEL_NAME, HIDDEN_SIZE, VIDEO_NUM_CLASSES, DROPOUT_P).to(device)
    
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
        "hidden_size": HIDDEN_SIZE,
        "num_classes": VIDEO_NUM_CLASSES,
        "dataset": DATASET_NAME,
        "optimizer": "AdamW",
        "resumed": RESUME_TRAINING,
        "use_wandb": USE_WANDB,
        "balance_dataset": BALANCE_DATASET,
        "use_default_split": USE_DEFAULT_SPLIT,
        "df_splitting": None if USE_DEFAULT_SPLIT else DF_SPLITTING,
        "apply_transformations": APPLY_TRANSFORMATIONS,
        "normalize": NORMALIZE,
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