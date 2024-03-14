from sklearn.metrics import recall_score, accuracy_score
from utils.utils import save_results, save_model, save_configurations
from tqdm import tqdm
import torch
import torch.nn as nn
import wandb
from datetime import datetime
import copy
from config import SAVE_MODELS, SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH


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
                print("--WANDB-- Temptative to resume a non-existing run. Starting a new one.")
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
        data_name = f"{config['scope']}_{config['architecture']}_{current_datetime_str}"

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

    total_step = len(train_loader)
    best_model = None
    best_accuracy = None
    for epoch in range(RESUME_EPOCH if resume else 0, config["epochs"]):
        model.train()
        epoch_tr_preds = torch.tensor([]).to(device)
        epoch_tr_labels = torch.tensor([]).to(device)
        for tr_i, tr_batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            if config["scope"] == "voice_emotion_recognition":
                tr_data, tr_labels = tr_batch['audio'], tr_batch['emotion'] # data = audio, labels = emotions
                tr_data = tr_data.to(device)
                tr_labels = tr_labels.to(device)

                tr_logits, tr_outputs = model(tr_data)  # Prediction

                # Multiclassification loss considering all classes
                tr_epoch_loss = criterion(tr_logits, tr_labels)

            optimizer.zero_grad()
            tr_epoch_loss.backward()
            optimizer.step()

            with torch.no_grad():
                tr_preds = torch.argmax(tr_outputs, -1).detach()
                epoch_tr_preds = torch.cat((epoch_tr_preds, tr_preds), 0)
                epoch_tr_labels = torch.cat((epoch_tr_labels, tr_labels), 0)

                tr_accuracy = accuracy_score(
                epoch_tr_labels.cpu().numpy(), epoch_tr_preds.cpu().numpy()) * 100
                tr_recall = recall_score(
                    epoch_tr_labels.cpu().numpy(), epoch_tr_preds.cpu().numpy(), average='macro', zero_division=0) * 100

                if (tr_i+1) % 5 == 0:
                    print('Training -> Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%'
                            .format(epoch+1, config["epochs"], tr_i+1, total_step, tr_epoch_loss, tr_accuracy, tr_recall))

        if config["use_wandb"]:
            wandb.log({"Training Loss": tr_epoch_loss.item()})
            wandb.log({"Training Accuracy": tr_accuracy})
            wandb.log({"Training Recall": tr_recall})

        model.eval()
        with torch.no_grad():
            epoch_val_preds = torch.tensor([]).to(device)
            epoch_val_labels = torch.tensor([]).to(device)
            for _, val_batch in enumerate(val_loader):
                if config["scope"] == "voice_emotion_recognition":
                    val_data, val_labels = val_batch['audio'], val_batch['emotion'] # data = audio, labels = emotions
                    val_data = val_data.to(device)
                    val_labels = val_labels.to(device)

                    val_logits, val_outputs = model(val_data)
                    val_preds = torch.argmax(val_outputs, -1).detach()
                    epoch_val_preds = torch.cat((epoch_val_preds, val_preds), 0)
                    epoch_val_labels = torch.cat((epoch_val_labels, val_labels), 0)

                    # Multiclassification loss considering all classes
                    val_epoch_loss = criterion(val_logits, val_labels)

            val_accuracy = accuracy_score(
                epoch_val_labels.cpu().numpy(), epoch_val_preds.cpu().numpy()) * 100
            val_recall = recall_score(epoch_val_labels.cpu().numpy(
            ), epoch_val_preds.cpu().numpy(), average='macro', zero_division=0) * 100
            if config["use_wandb"]:
                wandb.log({"Validation Loss": val_epoch_loss.item()})
                wandb.log({"Validation Accuracy": val_accuracy})
                wandb.log({"Validation Recall": val_recall})
            print('Validation -> Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%'
                  .format(epoch+1, config["epochs"], val_epoch_loss, val_accuracy, val_recall))

            if best_accuracy is None or val_accuracy < best_accuracy:
                best_accuracy = val_accuracy
                best_model = copy.deepcopy(model)
            current_results = {
                'epoch': epoch+1,
                'validation_loss': val_epoch_loss.item(),
                'training_loss': tr_epoch_loss.item(),
                'validation_accuracy': val_accuracy,
                'training_accuracy': tr_accuracy,
                'validation_recall': val_recall,
                'training_recall': tr_recall
            }
            if SAVE_RESULTS:
                save_results(data_name, current_results)
            if SAVE_MODELS:
                save_model(data_name, model, epoch)
            if epoch == config["epochs"]-1 and SAVE_MODELS:
                save_model(data_name, best_model, epoch=None, is_best=True)

        scheduler.step()