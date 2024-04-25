from config import BATCH_SIZE, SAVE_MODELS, SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH, LENGTH
from utils.utils import save_results, save_model, save_configurations
from datetime import datetime
from tqdm import tqdm
import torch
import wandb
import copy


def train_eval_loop(device,
                    train_loader: torch.utils.data.DataLoader,
                    val_loader: torch.utils.data.DataLoader,
                    model,
                    config,
                    optimizer,
                    scheduler,
                    criterion,
                    resume=False):

    if resume:
        data_name = PATH_MODEL_TO_RESUME
        if config["use_wandb"]:
            runs = wandb.api.runs("mi_project", filters={"name": data_name})
            if runs:
                run_id = runs[0]["id"]
                wandb.init(
                    project="mi_project",
                    id=run_id,
                    resume="allow",
                )
            else:
                print(
                    "--WANDB-- Temptative to resume a non-existing run. Starting a new one.")
                wandb.init(
                    project="mi_project",
                    config=config,
                    resume=resume,
                    name=data_name
                )
    else:
        # Definition of the parameters to create folders where to save data (plots and models)
        current_datetime = datetime.now()
        current_datetime_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        data_name = f"{config['architecture']}_{current_datetime_str}"

        if SAVE_RESULTS:
            # Save configurations in JSON
            save_configurations(data_name, config)

        if config["use_wandb"]:
            wandb.init(
                project="mi_project",
                config=config,
                resume=resume,
                name=data_name
            )

    for epoch in range(RESUME_EPOCH if resume else 0, config["epochs"]):

        model.train()
        losses = []

        for tr_i, tr_batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            src, target = tr_batch["ppg"], tr_batch["valence"]

            optimizer.zero_grad()

            src = src.float().to(device)
            target = target.to(device)
            
            hidden, cell = model.encoder(src)

            outputs = []
            for i in range(target.shape[1]):  # Iterate over sequence length
              curr_target = target[:, i].view(-1, 1, 1)
              output, hidden, cell = model.decoder(curr_target, hidden, cell)
              outputs.append(output)

            # Stack predictions and calculate loss
            predictions = torch.stack(outputs, 1)
            loss = criterion(predictions.view(-1, LENGTH), target)
            losses.append(loss.item()) 

            loss.backward()
            optimizer.step()

        print(f"Train | Epoch {epoch} | Loss: {torch.tensor(losses).mean():.4f}")
    
        model.eval()
        losses = []
        predictions = []
        targets = []

        with torch.no_grad():  # Disable gradient calculation for efficiency
            for val_i, val_batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
                src, target = val_batch["ppg"], val_batch["valence"]

                src = src.float().to(device)
                target = target.to(device)

                # Pass through encoder
                hidden, cell = model.encoder(src)

                # Decoder inference without teacher forcing
                outputs = []
                for i in range(target.shape[1]):  # Iterate over sequence length
                    curr_src = src[:, i:i+1].view(-1, 1, 1)
                    # print(f"Curr src shape: {curr_src.shape}")
                    output, hidden, cell = model.decoder(curr_src, hidden, cell)  # Use only current input
                    outputs.append(output)

                # Stack predictions and calculate loss
                predictions.append(torch.stack(outputs, 1).squeeze(1))  # Squeeze for regression
                targets.append(target)
                loss = criterion(torch.stack(outputs, 1).view(-1, LENGTH), target)
                losses.append(loss.item())

        print(f"Validation | Epoch {epoch} | Loss: {torch.tensor(losses).mean():.4f}")

