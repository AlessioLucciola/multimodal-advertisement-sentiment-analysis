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
    WAVELET_STEP,
    LSTM_HIDDEN,
    DROPOUT_P
)
from dataloaders.CEAP_dataloader import CEAPDataLoader
from models.EmotionNetCEAP import EmotionNet, Encoder, Decoder
from train.loops.train_loop_emotion_CEAP import train_eval_loop
from utils.utils import select_device, set_seed
import torch.nn as nn


def main():
    set_seed(RANDOM_SEED)
    device = select_device()
    loader = CEAPDataLoader(batch_size=BATCH_SIZE)

    train_loader = loader.get_train_dataloader()
    val_loader = loader.get_val_dataloader()

    input_dim = 1
    output_dim = 3
    encoder_embedding_dim = 1
    decoder_embedding_dim = 1
    hidden_dim = LSTM_HIDDEN
    n_layers = 2
    encoder_dropout = DROPOUT_P
    decoder_dropout = DROPOUT_P

    encoder = Encoder(
        input_dim,
        encoder_embedding_dim,
        hidden_dim,
        n_layers,
        encoder_dropout,
    )

    decoder = Decoder(
        output_dim,
        decoder_embedding_dim,
        hidden_dim,
        n_layers,
        decoder_dropout,
    )

    model = EmotionNet(encoder, decoder).to(device)

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)


    model.apply(init_weights)
    # if RESUME_TRAINING:
    #     model.load_state_dict(torch.load(
        #         f"{PATH_TO_SAVE_RESULTS}/{PATH_MODEL_TO_RESUME}/models/mi_project_{RESUME_EPOCH}.pt"))
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR,
                                 weight_decay=REG)

    # scheduler = torch.optim.lr_scheduler.StepLR(
    #         optimizer=optimizer,
    #         step_size=20, 
    #         gamma=0.1)

    scheduler = None

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
                    model=model,
                    config=config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    resume=RESUME_TRAINING)


if __name__ == "__main__":
    main()
