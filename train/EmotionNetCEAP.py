import torch
from config import *
from dataloaders.CEAP_dataloader import CEAPDataLoader
from dataloaders.DEAP_dataloader import DEAPDataLoader
from models.EmotionNetCEAP import EmotionNet, Encoder, Decoder
from models.EmotionNetDEAP import EmotionNet as EmotionNetDEAP
from train.loops.train_loop_emotion_CEAP import train_eval_loop as train_eval_loop_CEAP
from train.loops.train_loop_emotion_DEAP import train_eval_loop as train_eval_loop_DEAP
from utils.utils import select_device, set_seed
import torch.nn as nn

MODEL_NAME = "EmotionNetDEAP"

def get_model_and_dataloader():
    input_dim = LENGTH // WAVELET_STEP if WT else 1
    hidden_dim = LSTM_HIDDEN
    n_layers = LSTM_LAYERS

    if MODEL_NAME == "EmotionNetCEAP":
        loader = CEAPDataLoader(batch_size=BATCH_SIZE)

        output_dim = 3
        encoder_embedding_dim = LENGTH // WAVELET_STEP if WT else 1
        decoder_embedding_dim = LENGTH // WAVELET_STEP if WT else 1
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

        model = EmotionNet(encoder, decoder).to(select_device())

        def init_weights(m):
            for name, param in m.named_parameters():
                nn.init.uniform_(param.data, -0.08, 0.08)

        model.apply(init_weights)
        return model, loader

    elif MODEL_NAME == "EmotionNetDEAP":
        loader = DEAPDataLoader(batch_size=BATCH_SIZE)
        model = EmotionNetDEAP(input_size=input_dim, num_layers=LSTM_LAYERS, hidden_size=LSTM_HIDDEN, dropout_p=DROPOUT_P, num_classes=EMOTION_NUM_CLASSES).to(select_device())
        return model, loader
    else:
        raise ValueError(f"{MODEL_NAME} model not supported")

def main():
    set_seed(RANDOM_SEED)
    device = select_device()
    model, loader = get_model_and_dataloader()
    if RESUME_TRAINING:
        model.load_state_dict(torch.load(
                f"{PATH_TO_SAVE_RESULTS}/{PATH_MODEL_TO_RESUME}/models/mi_project_{RESUME_EPOCH}.pt"))

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR,
                                 weight_decay=REG)

    scheduler = None

    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.NLLLoss()
    # criterion = torch.nn.MSELoss()

    config = {
        "architecture": "EmotionNet - LSTM Seq2Seq",
        "scope": "EmotionNet",
        "learning_rate": LR,
        "epochs": N_EPOCHS,
        "reg": REG,
        "batch_size": BATCH_SIZE,
        "num_classes": EMOTION_NUM_CLASSES,
        "dataset": "CEAP",
        "optimizer": "Adam",
        "resumed": RESUME_TRAINING,
        "use_wandb": USE_WANDB,
        "balance_dataset": BALANCE_DATASET,
        "limit": LIMIT,
        "length": LENGTH,
        "step": STEP,
        "dropout_p": DROPOUT_P,
        "wavelet_step": WAVELET_STEP,
        "lstm_config": {
            "num_hidden": LSTM_HIDDEN,
            "num_layers": LSTM_LAYERS,
        }

    }
    if MODEL_NAME == "EmotionNetCEAP":
        train_eval_loop_CEAP(device=device,
                        train_loader=loader.get_train_dataloader(),
                        val_loader=loader.get_val_dataloader(),
                        model=model,
                        config=config,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        criterion=criterion,
                        resume=RESUME_TRAINING)
    elif MODEL_NAME == "EmotionNetDEAP":
        train_eval_loop_DEAP(device=device,
                        train_loader=loader.get_train_dataloader(),
                        val_loader=loader.get_val_dataloader(),
                        model=model,
                        config=config,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        criterion=criterion,
                        resume=RESUME_TRAINING)
    else:
        raise ValueError(f"{MODEL_NAME} model not supported")


if __name__ == "__main__":
    main()
