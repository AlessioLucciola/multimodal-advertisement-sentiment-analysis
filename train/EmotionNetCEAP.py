import torch
from config import (
    ADD_NOISE,
    AUGMENTATION_SIZE,
    BALANCE_DATASET,
    BATCH_SIZE,
    DROPOUT_P,
    EMOTION_NUM_CLASSES,
    LENGTH,
    LIMIT,
    LR,
    MESSAGE,
    N_EPOCHS,
    RANDOM_SEED,
    REG,
    RESUME_TRAINING,
    STEP,
    T_DIM_FFW,
    T_ENC_LAYERS,
    T_HEAD,
    T_KERN,
    T_MAXPOOL,
    T_STRIDE,
    USE_WANDB,
    WAVELET_STEP
)
from dataloaders.CEAP_dataloader import CEAPDataLoader
from models.EmotionNetCEAP import EmotionNet
from train.loops.train_loop_emotion_CEAP import train_eval_loop
from utils.utils import select_device, set_seed


def main():
    set_seed(RANDOM_SEED)
    device = select_device()
    loader = CEAPDataLoader(batch_size=BATCH_SIZE)

    train_loader = loader.get_train_dataloader()
    val_loader = loader.get_val_dataloader()

    model = EmotionNet(dropout=DROPOUT_P).to(device)

    # if RESUME_TRAINING:
    #     model.load_state_dict(torch.load(
        #         f"{PATH_TO_SAVE_RESULTS}/{PATH_MODEL_TO_RESUME}/models/mi_project_{RESUME_EPOCH}.pt"))
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR,
                                 weight_decay=REG)

    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=20, 
            gamma=0.1)

    scheduler = None

    criterion = torch.nn.MSELoss()

    config = {
        "architecture": "EmotionNet - CNN + LSTM",
        "scope": "EmotionNet",
        "learning_rate": LR,
        "epochs": N_EPOCHS,
        "reg": REG,
        "batch_size": BATCH_SIZE,
        "num_classes": EMOTION_NUM_CLASSES,
        "dataset": "GREX",
        "optimizer": "AdamW",
        "resumed": RESUME_TRAINING,
        "use_wandb": USE_WANDB,
        "balance_dataset": BALANCE_DATASET,
        "limit": LIMIT,
        "length": LENGTH,
        "step": STEP,
        "dropout_p": DROPOUT_P,
        "agumented_data": AUGMENTATION_SIZE,
        "add_noise": ADD_NOISE,
        "message": MESSAGE,
        "wavelet_step": WAVELET_STEP,
        "transformer_config": {
            "num_heads": T_HEAD,
            "num_encoder_layers": T_ENC_LAYERS,
            "num_dims_ffw": T_DIM_FFW,
            "kernel, stride": (T_KERN, T_STRIDE),
            "maxpool": T_MAXPOOL
            }

    }

    train_eval_loop(device=device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model=model,
                    config=config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    resume=RESUME_TRAINING)


if __name__ == "__main__":
    main()
