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
from dataloaders.GREX_dataloader import GREXDataLoader
from models.EmotionNetCL import EmotionNet
from train.loops.train_loop_emotion_single import train_eval_loop
from utils.utils import select_device, set_seed


def main():
    set_seed(RANDOM_SEED)
    device = select_device()
    loader = GREXDataLoader(batch_size=BATCH_SIZE)

    train_loader = loader.get_train_dataloader()
    val_loader = loader.get_val_dataloader()

    model_aro = EmotionNet(num_classes=EMOTION_NUM_CLASSES,
                           dropout=DROPOUT_P).to(device)
    model_val = EmotionNet(num_classes=EMOTION_NUM_CLASSES,
                           dropout=DROPOUT_P).to(device)
    model = [model_aro, model_val]

    # if RESUME_TRAINING:
    #     model.load_state_dict(torch.load(
    #         f"{PATH_TO_SAVE_RESULTS}/{PATH_MODEL_TO_RESUME}/models/mi_project_{RESUME_EPOCH}.pt"))
    optimizer_aro = torch.optim.Adam(model_aro.parameters(),
                                     lr=LR,
                                     weight_decay=REG)

    optimizer_val = torch.optim.Adam(model_val.parameters(),
                                     lr=LR,
                                     weight_decay=REG)

    optimizer = [optimizer_aro, optimizer_val]

    scheduler_val = None

    scheduler_aro = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer_aro,
            step_size=20, 
            gamma=0.1)

    # schedulers = [scheduler_aro, scheduler_val]
    schedulers = [None, None]

    criterion = torch.nn.CrossEntropyLoss()

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
                    models=model,
                    config=config,
                    optimizers=optimizer,
                    schedulers=schedulers,
                    criterion=criterion,
                    scaler=None,
                    resume=RESUME_TRAINING)


if __name__ == "__main__":
    main()
