from config import BATCH_SIZE, SAVE_MODELS, SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH
from utils.utils import save_results, save_model, save_configurations, save_scaler
from torchmetrics import Accuracy, Recall, Precision, F1Score
from datetime import datetime
from tqdm import tqdm
import torch
import wandb


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
            if scaler is not None:
                save_scaler(data_name, scaler)

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

    model_aro, model_val = models
    optimizer_aro, optimizer_val = optimizers
    for epoch in range(RESUME_EPOCH if resume else 0, config["epochs"]):
        tr_cumulative_loss_val = 0
        tr_cumulative_loss_aro = 0
        val_cumulative_loss_val = 0
        val_cumulative_loss_aro = 0
        tr_step = 0
        val_step = 0

        model_val.train()
        model_aro.train()
        epoch_tr_preds_val = torch.tensor([]).to(device)
        epoch_tr_labels_val = torch.tensor([]).to(device)
        epoch_tr_preds_aro = torch.tensor([]).to(device)
        epoch_tr_labels_aro = torch.tensor([]).to(device)

        for tr_i, tr_batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            tr_data, tr_spatial, tr_labels_valence, tr_labels_arousal = tr_batch["ppg"], tr_batch[
                "ppg_spatial_features"], tr_batch["valence"], tr_batch["arousal"]

            tr_data = tr_data.float().to(device)
            tr_labels_valence = tr_labels_valence.to(device)
            tr_labels_arousal = tr_labels_arousal.to(device)
            tr_spatial = tr_spatial.float().to(device)

            tr_outputs_val = model_val(tr_data, tr_spatial)
            tr_outputs_aro = model_aro(tr_data, tr_spatial)

            epoch_tr_preds_val = torch.cat(
                (epoch_tr_preds_val, tr_outputs_val), 0)
            epoch_tr_labels_val = torch.cat(
                (epoch_tr_labels_val, tr_labels_valence), 0)

            epoch_tr_preds_aro = torch.cat(
                (epoch_tr_preds_aro, tr_outputs_aro), 0)
            epoch_tr_labels_aro = torch.cat(
                (epoch_tr_labels_aro, tr_labels_arousal), 0)

            tr_epoch_loss_val = criterion(
                tr_outputs_val, tr_labels_valence)

            tr_epoch_loss_aro = criterion(
                tr_outputs_aro, tr_labels_arousal)

            tr_cumulative_loss_val += tr_epoch_loss_val
            tr_cumulative_loss_aro += tr_epoch_loss_aro

            optimizer_val.zero_grad()
            tr_epoch_loss_val.backward()
            optimizer_val.step()

            optimizer_aro.zero_grad()
            tr_epoch_loss_aro.backward()
            optimizer_aro.step()

            # for name, param in model_aro.named_parameters():
            #     print(
            #         f"Gradients for the Arousal classifier: {name, param.grad}")
            # for name, param in model_val.named_parameters():
            #     print(
            #         f"Gradients for the Valence classifier: {name, param.grad}")

            tr_step += 1

        with torch.no_grad():
            loss = (tr_cumulative_loss_val / tr_step +
                    tr_cumulative_loss_aro / tr_step) / 2

            tr_preds_val = torch.argmax(epoch_tr_preds_val, -1).detach()
            tr_preds_aro = torch.argmax(epoch_tr_preds_aro, -1).detach()

            tr_accuracy_val = accuracy_metric(
                tr_preds_val, epoch_tr_labels_val) * 100
            tr_accuracy_aro = accuracy_metric(
                tr_preds_aro, epoch_tr_labels_aro) * 100

            tr_recall_val = recall_metric(
                tr_preds_val, epoch_tr_labels_val) * 100
            tr_recall_aro = recall_metric(
                tr_preds_aro, epoch_tr_labels_aro) * 100

        if config["use_wandb"]:
            wandb.log(
                {"Training Loss": loss.item()})
            wandb.log({"Training Accuracy": (
                (tr_accuracy_aro + tr_accuracy_val) / 2).item()})
            wandb.log({"Training Recall": (
                (tr_recall_aro + tr_recall_val) / 2).item()})

        model_val.eval()
        model_aro.eval()
        with torch.no_grad():
            epoch_val_preds_val = torch.tensor([]).to(device)
            epoch_val_labels_val = torch.tensor([]).to(device)
            epoch_val_preds_aro = torch.tensor([]).to(device)
            epoch_val_labels_aro = torch.tensor([]).to(device)
            for _, val_batch in enumerate(val_loader):
                val_data, val_spatial, val_labels_valence, val_labels_arousal = val_batch["ppg"], val_batch[
                    "ppg_spatial_features"], val_batch["valence"], val_batch["arousal"]
                val_data = val_data.float().to(device)
                val_spatial = val_spatial.float().to(device)
                val_labels_valence = val_labels_valence.to(device)
                val_labels_arousal = val_labels_arousal.to(device)

                val_outputs_val = model_val(val_data, val_spatial)
                val_outputs_aro = model_aro(val_data, val_spatial)

                val_preds_val = torch.argmax(val_outputs_val, -1).detach()
                epoch_val_preds_val = torch.cat(
                    (epoch_val_preds_val, val_preds_val), 0)
                epoch_val_labels_val = torch.cat(
                    (epoch_val_labels_val, val_labels_valence), 0)

                val_preds_aro = torch.argmax(val_outputs_aro, -1).detach()
                epoch_val_preds_aro = torch.cat(
                    (epoch_val_preds_aro, val_preds_aro), 0)
                epoch_val_labels_aro = torch.cat(
                    (epoch_val_labels_aro, val_labels_arousal), 0)

                val_epoch_loss_val = criterion(
                    val_outputs_val, val_labels_valence)
                val_epoch_loss_aro = criterion(
                    val_outputs_aro, val_labels_arousal)

                val_cumulative_loss_val += val_epoch_loss_val
                val_cumulative_loss_aro += val_epoch_loss_aro
                val_step += 1

            val_accuracy_val = accuracy_metric(
                epoch_val_preds_val, epoch_val_labels_val) * 100
            val_recall_val = recall_metric(
                epoch_val_preds_val, epoch_val_labels_val) * 100

            val_accuracy_aro = accuracy_metric(
                epoch_val_preds_aro, epoch_val_labels_aro) * 100
            val_recall_aro = recall_metric(
                epoch_val_preds_aro, epoch_val_labels_aro) * 100

            val_loss = (val_cumulative_loss_val / val_step +
                        val_cumulative_loss_aro / val_step) / 2

            if config["use_wandb"]:
                wandb.log(
                    {"Validation Loss": val_loss.item()})
                wandb.log({"Validation Accuracy": (
                    (val_accuracy_aro + val_accuracy_val) / 2).item()})
                wandb.log({"Validation Recall": (
                    (val_recall_aro + val_recall_val) / 2).item()})

            # if best_accuracy is None or val_accuracy < best_accuracy:
            #     best_accuracy = val_accuracy
            #     best_model = copy.deepcopy(model)
            # current_results = {
            #     'epoch': epoch+1,
            #     'training_loss': tr_epoch_loss.item(),
            #     'training_accuracy': tr_accuracy.item(),
            #     'training_recall': tr_recall.item(),
            #     'training_precision': tr_precision.item(),
            #     'training_f1': tr_f1.item(),
            #     'training_auroc': tr_auroc.item(),
            #     'validation_loss': val_epoch_loss.item(),
            #     'validation_accuracy': val_accuracy.item(),
            #     'validation_recall': val_recall.item(),
            #     'validation_precision': val_precision.item(),
            #     'validation_f1': val_f1.item(),
            #     'validation_auroc': val_auroc.item()
            # }
            # if SAVE_RESULTS:
            #     save_results(data_name, current_results)
            # if SAVE_MODELS:
            #     save_model(data_name, model, epoch)
            # if epoch == config["epochs"]-1 and SAVE_MODELS:
            #     save_model(data_name, best_model, epoch=None, is_best=True)

        if scheduler is not None:
            scheduler.step()
        print(
            f"Training -> Epoch[{epoch+1}/{config['epochs']}], Loss: {loss:.4f}), Accuracy: {((tr_accuracy_aro + tr_accuracy_val) / 2): .4f} % , Recall: {((tr_recall_aro + tr_recall_val) / 2): .4f} %")

        print(
            f"Validation -> Epoch[{epoch+1}/{config['epochs']}], Loss: {val_loss:.4f}), Accuracy: {((val_accuracy_aro+ val_accuracy_val) / 2): .4f} % , Recall: {((val_recall_aro+ val_recall_val) / 2): .4f} %")
        print("-"*50)
