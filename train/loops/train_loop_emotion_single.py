from config import BATCH_SIZE, SAVE_MODELS, SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH
from utils.utils import save_results, save_model, save_configurations, save_scaler
from torchmetrics import Accuracy, Recall, Precision, F1Score
from datetime import datetime
from tqdm import tqdm
import torch
import wandb
import copy


def train_eval_loop(device,
                    train_loader: torch.utils.data.DataLoader,
                    val_loader: torch.utils.data.DataLoader,
                    models,
                    config,
                    optimizers,
                    scheduler,
                    criterion,
                    scaler,
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
    best_model = None
    best_accuracy = None

    accuracy_metric = Accuracy(
        task="multiclass", num_classes=config['num_classes']).to(device)
    recall_metric = Recall(
        task="multiclass", num_classes=config['num_classes'], average='macro').to(device)

    model = models[0]
    optimizer = optimizers[0]
    for epoch in range(RESUME_EPOCH if resume else 0, config["epochs"]):
        tr_cumulative_loss = 0
        val_cumulative_loss = 0
        tr_step = 0
        val_step = 0

        model.train()
        epoch_tr_preds = torch.tensor([]).to(device)
        epoch_tr_labels = torch.tensor([]).to(device)

        for tr_i, tr_batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            tr_data, tr_spatial, tr_labels = tr_batch["ppg"], tr_batch[
                "ppg_spatial_features"], tr_batch["valence"]

            tr_data = tr_data.float().to(device)
            tr_labels = tr_labels.to(device)
            tr_spatial = tr_spatial.float().to(device)

            tr_outputs = model(tr_data, tr_spatial)

            epoch_tr_preds = torch.cat(
                (epoch_tr_preds, tr_outputs), 0)
            epoch_tr_labels = torch.cat(
                (epoch_tr_labels, tr_labels), 0)

            tr_epoch_loss = criterion(
                tr_outputs, tr_labels)

            tr_cumulative_loss += tr_epoch_loss

            optimizer.zero_grad()
            tr_epoch_loss.backward()
            optimizer.step()

            tr_step += 1

        with torch.no_grad():
            tr_preds = torch.argmax(epoch_tr_preds, -1)
            tr_accuracy = accuracy_metric(
                tr_preds, epoch_tr_labels) * 100
            tr_recall = recall_metric(
                tr_preds, epoch_tr_labels) * 100

        tr_cumulative_loss /= tr_step
        if config["use_wandb"]:
            wandb.log(
                {"Training Loss": tr_cumulative_loss.item()})
            wandb.log({"Training Accuracy": tr_accuracy.item()})
            wandb.log({"Training Recall": tr_recall.item()})

        model.eval()
        with torch.no_grad():
            epoch_val_preds = torch.tensor([]).to(device)
            epoch_val_labels = torch.tensor([]).to(device)
            for _, val_batch in enumerate(val_loader):
                val_data, val_spatial, val_labels = val_batch["ppg"], val_batch[
                    "ppg_spatial_features"], val_batch["valence"]

                val_data = val_data.float().to(device)
                val_spatial = val_spatial.float().to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_data, val_spatial)

                val_preds = torch.argmax(val_outputs, -1).detach()

                epoch_val_preds = torch.cat((epoch_val_preds, val_preds), 0)
                epoch_val_labels = torch.cat((epoch_val_labels, val_labels), 0)

                val_epoch_loss = criterion(val_outputs, val_labels)

                val_cumulative_loss += val_epoch_loss
                val_step += 1

            val_accuracy = accuracy_metric(
                epoch_val_preds, epoch_val_labels) * 100
            val_recall = recall_metric(
                epoch_val_preds, epoch_val_labels) * 100

            val_cumulative_loss /= val_step
            if config["use_wandb"]:
                wandb.log(
                    {"Validation Loss": val_cumulative_loss.item()})
                wandb.log({"Validation Accuracy":
                           val_accuracy.item()})
                wandb.log({"Validation Recall": val_recall.item()})

            if best_accuracy is None or val_accuracy < best_accuracy:
                best_accuracy = val_accuracy
                best_model = copy.deepcopy(model)

            current_results = {
                'epoch': epoch+1,
                'training_loss': tr_cumulative_loss.item(),
                'training_accuracy': tr_accuracy.item(),
                'training_recall': tr_recall.item(),
                # 'training_precision': tr_precision.item(),
                # 'training_f1': tr_f1.item(),
                # 'training_auroc': tr_auroc.item(),
                'validation_loss': val_cumulative_loss.item(),
                'validation_accuracy': val_accuracy.item(),
                'validation_recall': val_recall.item(),
                # 'validation_precision': val_precision.item(),
                # 'validation_f1': val_f1.item(),
                # 'validation_auroc': val_auroc.item()
            }
            if SAVE_RESULTS:
                save_results(data_name, current_results)
            if SAVE_MODELS:
                save_model(data_name, model, epoch)
            if epoch == config["epochs"]-1 and SAVE_MODELS:
                save_model(data_name, best_model, epoch=None, is_best=True)

        if scheduler is not None:
            scheduler.step()

        print(
            f"Training -> Epoch[{epoch+1}/{config['epochs']}], Loss: {tr_cumulative_loss:.4f}), Accuracy: {tr_accuracy: .4f} % , Recall: {tr_recall: .4f} %")

        print(
            f"Validation -> Epoch[{epoch+1}/{config['epochs']}], Loss: {val_cumulative_loss:.4f}), Accuracy: {val_accuracy: .4f} % , Recall: {val_recall: .4f} %")
        print("-"*50)
