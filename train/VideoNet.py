import torch
from config import USE_POSITIVE_NEGATIVE_LABELS, OVERLAP_SUBJECTS_FRAMES, PRELOAD_FRAMES, VIDEO_METADATA_FRAMES_CSV, FRAMES_FILES_DIR, VIDEO_DATASET_NAME, DATASET_NAME, RANDOM_SEED, USE_WANDB, LIMIT, MODEL_NAME, BATCH_SIZE, LR, N_EPOCHS, VIDEO_METADATA_CSV, REG, NUM_CLASSES, DROPOUT_P, RESUME_TRAINING, PATH_TO_SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH, BALANCE_DATASET, APPLY_TRANSFORMATIONS, DF_SPLITTING, HIDDEN_SIZE, NORMALIZE, IMG_SIZE
from dataloaders.video_custom_dataloader import video_custom_dataloader
from train.loops.train_loop import train_eval_loop
from utils.utils import set_seed, select_device
from utils.video_utils import select_model
from shared.constants import video_cnn_models_list, video_vit_models_list

def main():
    set_seed(RANDOM_SEED)
    device = select_device()

    # Create custom dataloader
    custom_dataloader = video_custom_dataloader(csv_original_files=VIDEO_METADATA_CSV,
                                   csv_frames_files=VIDEO_METADATA_FRAMES_CSV,
                                   batch_size=BATCH_SIZE,
                                   frames_dir=FRAMES_FILES_DIR,
                                   seed=RANDOM_SEED,
                                   limit=LIMIT,
                                   use_positive_negative_labels=USE_POSITIVE_NEGATIVE_LABELS,
                                   preload_frames=PRELOAD_FRAMES,
                                   apply_transformations=APPLY_TRANSFORMATIONS,
                                   balance_dataset=BALANCE_DATASET,
                                   normalize=NORMALIZE,
                                   )
    
    # Get train and validation dataloaders
    train_loader = custom_dataloader.get_train_dataloader()
    val_loader = custom_dataloader.get_val_dataloader()

    # Select model
    model = select_model(MODEL_NAME, HIDDEN_SIZE, NUM_CLASSES, DROPOUT_P).to(device)

    # Freeze layers for CNN models
    if video_cnn_models_list.count(MODEL_NAME) > 0:
        # Freeze all layers except the last one
        for p in model.parameters():
            p.requires_grad = False

        # LAYERS_TO_FINE_TUNE = 20
        # parameters = list(model.parameters())
        # for p in parameters[-LAYERS_TO_FINE_TUNE:]:
        #     p.requires_grad=True

        # Unfreeze all layers
        for p in model.classifier.parameters():
            p.requires_grad = True

    print(f"--Model-- Using {MODEL_NAME} model")

    if RESUME_TRAINING:
        model.load_state_dict(torch.load(
            f"{PATH_TO_SAVE_RESULTS}/{PATH_MODEL_TO_RESUME}/models/mi_project_{RESUME_EPOCH}.pt"))
        
    # Define optimizer, scheduler and criterion
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=LR, 
                                 weight_decay=REG)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=N_EPOCHS,
        eta_min=1e-2
        )                                
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
        "video_metadata_csv": VIDEO_METADATA_CSV,
        "optimizer": "AdamW",
        "resumed": RESUME_TRAINING,
        "use_wandb": USE_WANDB,
        "balance_dataset": BALANCE_DATASET,
        "df_splitting": DF_SPLITTING,
        "img_size": IMG_SIZE,
        "use_positive_negative_labels": USE_POSITIVE_NEGATIVE_LABELS,
        "overlap_subjects_frames": OVERLAP_SUBJECTS_FRAMES,
        "preload_frames": PRELOAD_FRAMES,
        "apply_transformations": APPLY_TRANSFORMATIONS,
        "normalize": NORMALIZE,
        "limit": LIMIT,
        "dropout_p": DROPOUT_P,
    }

    # Train and evaluate the model
    train_eval_loop(device=device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model=model,
                    config=config,
                    optimizer=optimizer,
                    scaler=None,
                    scheduler=scheduler,
                    criterion=criterion,
                    resume=RESUME_TRAINING)
    
if __name__ == "__main__":
    main()