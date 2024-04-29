from argparse import ArgumentParser
import itertools
import json
import os
from datetime import datetime

import torch
from tqdm import tqdm
import wandb

from dataloaders.ravdess_custom_dataloader import ravdess_custom_dataloader
from train.loops.train_loop import train_eval_loop
from config import NORMALIZE, RANDOM_SEED, LIMIT, MODEL_NAME, BATCH_SIZE, LR, N_EPOCHS, METADATA_CSV, REG, VIDEO_NUM_CLASSES, DROPOUT_P, RESUME_TRAINING, BALANCE_DATASET, DATASET_NAME, USE_DEFAULT_SPLIT, APPLY_TRANSFORMATIONS, DF_SPLITTING, HIDDEN_SIZE
from utils.utils import set_seed, select_device
from utils.video_utils import select_model

device = select_device()


def init_with_parsed_arguments():
    set_seed(RANDOM_SEED)
    parser = ArgumentParser()
    parser.add_argument("--apply_transformations", action="store_true", default=False)
    parser.add_argument("--balance_dataset", action="store_true", default=False)
    parser.add_argument("--use_default_split", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=False)

    # REQUIRED: resnet18, resnet34, resnet50, resnet101, densenet121, custom-cnn, vit-pretrained
    parser.add_argument("--architecture", type=str)
    parser.add_argument("--dataset-limit", type=float, default=LIMIT)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--reg", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)

    # If True, it will not use wandb for logging
    parser.add_argument("--no-wandb", action="store_true", default=False)

    # If True, will reset the combinations tried for the current architecture
    parser.add_argument("--force-reset", action="store_true", default=False)

    parser.add_argument("--message", type=str, default=None)

    args = parser.parse_args()
    assert args.architecture is not None, "You must specify an architecture"

    print(f"Parsed arguments are {args}")
    print(f"Vars parsed arguments are {vars(args)}")

    kwargs = vars(args)

    config = {
        "architecture": "HT_VideoNet_" + kwargs.get("architecture"),
        "scope": "VideoNet",
        "learning_rate": LR if kwargs.get("lr") is None else kwargs.get("lr"),
        "epochs": N_EPOCHS if kwargs.get("epochs") is None else kwargs.get("epochs"),
        "reg": REG if kwargs.get("reg") is None else kwargs.get("reg"),
        "batch_size": BATCH_SIZE if kwargs.get("batch_size") is None else kwargs.get("batch_size"),
        "hidden_size": HIDDEN_SIZE,
        "num_classes": VIDEO_NUM_CLASSES,
        "dataset": DATASET_NAME,
        "optimizer": "AdamW",
        "resumed": RESUME_TRAINING,
        "use_wandb": True if not kwargs.get("no_wandb") else False,
        "balance_dataset": BALANCE_DATASET if kwargs.get("balance_dataset") is None else kwargs.get("balance_dataset"),
        "use_default_split": USE_DEFAULT_SPLIT if kwargs.get("use_default_split") is None else kwargs.get("use_default_split"),
        "df_splitting": None if kwargs.get("use_default_split") is None else DF_SPLITTING,
        "apply_transformations": APPLY_TRANSFORMATIONS if kwargs.get("apply_transformations") is None else kwargs.get("apply_transformations"),
        "normalize": NORMALIZE if kwargs.get("normalize") is None else kwargs.get("normalize"),
        "limit": LIMIT if kwargs.get("dataset_limit") is None else kwargs.get("dataset_limit"),
        "dropout_p": DROPOUT_P if kwargs.get("dropout") is None else kwargs.get("dropout"),
        # NEW
        "hparam_tuning": True if (kwargs.get("reg") is None and kwargs.get("dropout") is None) else False,
        "force_reset": kwargs.get("force_reset"),
        "message": kwargs.get("message") if kwargs.get("message") is not None else None
    }

    train_loader, val_loader = build_dataloaders(**config)
    if args.reg is not None and args.dropout is not None:
        print(f"----REG AND DROPOUT_P ARE NOT NONE, NOT DOING HPARAMS TUNING----")
        init_run(train_loader=train_loader,
                 val_loader=val_loader,
                 **config)
    else:
        hparams_tuning(train_loader=train_loader,
                       val_loader=val_loader,
                       **config)


def hparams_tuning(train_loader, val_loader, **hparams):
    hparams_space = {
        # "reg": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        # "dropout_p": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        "reg": [0.02, 0.03, 0.04],
        "dropout_p": [0.2, 0.3, 0.4]
    }
    combinations = list(itertools.product(*hparams_space.values()))

    if hparams["force_reset"]:
        combinations_tried = {}
    else:
        # Filter combinations that have already been tried
        if os.path.exists("./results/combinations.json"):
            with open("./results/combinations.json", "r") as f:
                combinations_tried = json.load(f)
        else:
            combinations_tried = {}

        print(f"COMBINATIONS TRIED: {combinations_tried}")

        curr_architecture = f"{hparams['architecture']}"
        if curr_architecture in combinations_tried:
            combinations = [
                combination for combination in combinations if list(combination) not in combinations_tried[curr_architecture]]
            print(
                f"----Found {len(combinations_tried[curr_architecture])} combinations already tried for {curr_architecture}, excluding them from the run! ----")
        else:
            combinations_tried[curr_architecture] = []

    print(f"Combinations are {combinations}")
    for combination in tqdm(combinations, "Hparams tuning"):
        hparams.update(dict(zip(hparams_space.keys(), combination)))
        # print(f"Hparams are {hparams}")
        init_run(train_loader=train_loader,
                 val_loader=val_loader,
                 **hparams)

        combinations_tried[curr_architecture].append(combination)
        with open("./results/combinations.json", "w") as f:
            json.dump(combinations_tried, f)


def init_run(train_loader, val_loader, **kwargs):
    model = get_model(**kwargs)

    # Definition of the parameters to create folders where to save data (plots and models)
    current_datetime = datetime.now()
    current_datetime_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    data_name = f"{kwargs['architecture']}_{current_datetime_str}"

    if kwargs["use_wandb"]:
        if wandb.run is not None:
            wandb.finish()
        wandb.init(
                project="mi_project",
                config=kwargs,
                resume=False,
                name=data_name
            )

    run_train_eval_loop(model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        **kwargs)


def get_model(**kwargs):
    architecture = kwargs.get("architecture")
    dropout_p = kwargs.get("dropout_p")

    model = select_model(MODEL_NAME, HIDDEN_SIZE, VIDEO_NUM_CLASSES, dropout_p).to(device)

    if architecture in ["resnet18", "resnet34", "resnet50", "resnet101", "densenet121"]:
        for p in model.parameters():
            p.requires_grad = False

        print(f"--Model-- Using {architecture} pretrained model")

        for p in model.classifier.parameters():
            p.requires_grad = True

    return model


def build_dataloaders(**args):
    fer_dataloader = ravdess_custom_dataloader(csv_file=METADATA_CSV,
                                   batch_size=args["batch_size"],
                                   seed=RANDOM_SEED,
                                   limit=args["limit"],
                                   apply_transformations=args["apply_transformations"],
                                   balance_dataset=args["balance_dataset"],
                                   use_default_split=args["use_default_split"],
                                   normalize=args["normalize"],
                                   )
    
    train_loader = fer_dataloader.get_train_dataloader()
    val_loader = fer_dataloader.get_val_dataloader()

    return train_loader, val_loader


def run_train_eval_loop(model, train_loader, val_loader, **kwargs):
    print(f"---CURRENT CONFIGURATION---\n{kwargs}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=kwargs["learning_rate"], weight_decay=kwargs["reg"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=kwargs["epochs"], eta_min=1e-5, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    train_eval_loop(device=device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model=model,
                    config=kwargs,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    scaler=None,
                    resume=RESUME_TRAINING)

if __name__ == "__main__":
    init_with_parsed_arguments()
