import torch
from config import PRELOAD_FRAMES, FRAMES_FILES_DIR, VIDEO_DATASET_NAME, DATASET_NAME, RANDOM_SEED, USE_WANDB, LIMIT, MODEL_NAME, BATCH_SIZE, LR, N_EPOCHS, VIDEO_METADATA_CSV, REG, NUM_CLASSES, DROPOUT_P, RESUME_TRAINING, PATH_TO_SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH, BALANCE_DATASET, APPLY_TRANSFORMATIONS, DF_SPLITTING, HIDDEN_SIZE, NORMALIZE, IMG_SIZE
from dataloaders.video_custom_dataloader import video_custom_dataloader
from train.loops.train_loop import train_eval_loop
from utils.utils import set_seed, select_device
from utils.video_utils import select_model

def main():
    set_seed(RANDOM_SEED)
    device = select_device()

    custom_dataloader = video_custom_dataloader(csv_file=VIDEO_METADATA_CSV,
                                   batch_size=BATCH_SIZE,
                                   frames_dir=FRAMES_FILES_DIR,
                                   seed=RANDOM_SEED,
                                   limit=LIMIT,
                                   preload_frames=PRELOAD_FRAMES,
                                   apply_transformations=APPLY_TRANSFORMATIONS,
                                   balance_dataset=BALANCE_DATASET,
                                   normalize=NORMALIZE,
                                   )
    
    train_loader = custom_dataloader.get_train_dataloader()
    val_loader = custom_dataloader.get_val_dataloader()
    
    model = select_model(MODEL_NAME, HIDDEN_SIZE, NUM_CLASSES, DROPOUT_P).to(device)
    
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
        "num_classes": NUM_CLASSES,
        "dataset": DATASET_NAME,
        "video_dataset": VIDEO_DATASET_NAME,
        "optimizer": "AdamW",
        "resumed": RESUME_TRAINING,
        "use_wandb": USE_WANDB,
        "balance_dataset": BALANCE_DATASET,
        "df_splitting": DF_SPLITTING,
        "img_size": IMG_SIZE,
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