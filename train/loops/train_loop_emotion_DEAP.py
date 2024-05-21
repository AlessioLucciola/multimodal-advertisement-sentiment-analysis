from config import SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH, SAVE_MODELS 
from utils.utils import save_results, save_configurations, save_model
from torchmetrics import Accuracy, Recall, F1Score
from datetime import datetime
from tqdm import tqdm
import torch
import wandb
import copy
import torch

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

    accuracy_metric = Accuracy(
        task="multiclass", num_classes=config['num_classes']).to(device)
    recall_metric = Recall(
        task="multiclass", num_classes=config['num_classes'], average='macro').to(device)
    f1_metric = F1Score(task="multiclass", num_classes=config['num_classes'], average='macro').to(device)

    val_accuracy_metric = Accuracy(
        task="multiclass", num_classes=config['num_classes']).to(device)
    val_recall_metric = Recall(
        task="multiclass", num_classes=config['num_classes'], average='macro').to(device)
    val_f1_metric = F1Score(task="multiclass", num_classes=config['num_classes'], average='macro').to(device)
    
    best_accuracy = 0.40
    best_model = None

    for epoch in range(RESUME_EPOCH if resume else 0, config["epochs"]):
        model.train()
        losses = []

        for tr_batch in tqdm(train_loader, desc="Training", leave=False):
            src, target = tr_batch["ppg"], tr_batch["valence"]

            src = src.float().to(device)
            target = target.long().to(device)

            optimizer.zero_grad()
            output = model(src).squeeze()

            loss = criterion(output, target)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

            # Calculate accuracy and recall
            with torch.no_grad():
                preds = output.argmax(dim=1)
                accuracy_metric.update(preds, target)
                f1_metric.update(preds, target)
                recall_metric.update(preds, target)

        if config["use_wandb"]:
            wandb.log(
                    {"Training Loss": torch.tensor(losses).mean()})
            wandb.log({"Training Accuracy": accuracy_metric.compute() * 100})
            wandb.log({"Training Recall": recall_metric.compute() * 100})
            wandb.log({"Training F1": f1_metric.compute() * 100})

        print(f"Train | Epoch {epoch} | Loss: {torch.tensor(losses).mean():.4f} | Accuracy: {(accuracy_metric.compute() * 100):.4f} | Recall: {(recall_metric.compute() * 100):.4f} | F1: {(f1_metric.compute() * 100):.4f}")
        
        model.eval()
        val_losses = []
        with torch.no_grad():  # Disable gradient calculation for efficiency
            for val_batch in tqdm(val_loader, desc="Validation", leave=False):
                src, target = val_batch["ppg"], val_batch["valence"]

                src = src.float().to(device)
                target = target.long().to(device)

                output = model(src).squeeze()
                loss = criterion(output, target)
                val_losses.append(loss.item())

                preds = output.argmax(dim=1)
                val_accuracy_metric.update(preds, target)
                val_recall_metric.update(preds, target)
                val_f1_metric.update(preds, target)

        if config["use_wandb"]:
            wandb.log(
                    {"Validation Loss": torch.tensor(val_losses).mean()})
            wandb.log({"Validation Accuracy": val_accuracy_metric.compute() * 100})
            wandb.log({"Validation Recall": val_recall_metric.compute() * 100})
            wandb.log({"Validation F1": val_f1_metric.compute() * 100})

        print(f"Validation | Epoch {epoch} | Loss: {torch.tensor(val_losses).mean():.4f} | Accuracy: {(val_accuracy_metric.compute() * 100):.4f} | Recall: {(val_recall_metric.compute() * 100):.4f} | F1: {(val_f1_metric.compute() * 100):.4f}")
        print("-" * 50)
    
        if val_accuracy_metric.compute() > best_accuracy and val_recall_metric.compute() > 0.4:
            print(f"Best model saved (accuracy: {val_accuracy_metric.compute()}, previous: {best_accuracy}")
            best_accuracy = val_accuracy_metric.compute()
            best_model = copy.deepcopy(model)
            if SAVE_MODELS:
              save_model(data_name, model, epoch)
        current_results = {
            'epoch': epoch+1,
            'training_loss': torch.tensor(losses).mean().item(),
            'training_accuracy': accuracy_metric.compute().item(),
            'training_recall': recall_metric.compute().item(),
            'training_f1': f1_metric.compute().item(),
            'validation_loss': torch.tensor(val_losses).mean().item(),
            'validation_accuracy': val_accuracy_metric.compute().item(),
            'validation_recall': val_recall_metric.compute().item(),
            'validation_f1': val_f1_metric.compute().item(),
        }
        if SAVE_RESULTS:
            save_results(data_name, current_results)
        if epoch == config["epochs"]-1 and SAVE_MODELS:
            save_model(data_name, best_model, epoch=None, is_best=True)
