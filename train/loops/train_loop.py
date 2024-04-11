from config import SAVE_MODELS, SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH
from utils.utils import save_results, save_model, save_configurations, save_scaler
from torchmetrics import Accuracy, Recall, Precision, F1Score, AUROC
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

    total_step = len(train_loader)
    best_model = None
    best_accuracy = None
    accuracy_metric = Accuracy(
        task="multiclass", num_classes=config['num_classes']).to(device)
    recall_metric = Recall(
        task="multiclass", num_classes=config['num_classes'], average='macro').to(device)
    precision_metric = Precision(
        task="multiclass", num_classes=config['num_classes'], average='macro').to(device)
    f1_metric = F1Score(
        task="multiclass", num_classes=config['num_classes'], average='macro').to(device)
    auroc_metric = AUROC(
        task="multiclass", num_classes=config['num_classes']).to(device)

    for epoch in range(RESUME_EPOCH if resume else 0, config["epochs"]):
        epoch_tr_accuracy = 0
        epoch_tr_recall = 0
        tr_cumulative_loss = 0
        val_cumulative_loss = 0
        tr_step = 0
        val_step = 0

        model.train()
        epoch_tr_preds = torch.tensor([]).to(device)
        epoch_tr_labels = torch.tensor([]).to(device)
        for tr_i, tr_batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            if config["scope"] == "AudioNet":
                # data = audio, labels = emotions
                tr_data, tr_labels = tr_batch['audio'], tr_batch['emotion']
            elif config["scope"] == "VideoNet":
                # data = pixel, labels = emotions
                tr_data, tr_labels = tr_batch[0], tr_batch[1]
            elif config["scope"] == "EmotionNet":
                tr_data, tr_spatial, tr_temp, tr_labels_valence, tr_labels_arousal = tr_batch["ppg"], tr_batch[
                    "ppg_spatial_features"], tr_batch["ppg_temporal_features"], tr_batch["valence"], tr_batch["arousal"]

                # tr_labels = torch.stack(tr_labels).transpose(
                #     0, 1)
                # tr_features = tr_features.float().to(device)
            else:
                raise ValueError(f"Invalid scope: {config['scope']}")

            tr_data = tr_data.float().to(device)
            tr_labels_valence = tr_labels_valence.float().to(device)
            tr_labels_arousal = tr_labels_arousal.float().to(device)
            tr_spatial = tr_spatial.float().to(device)
            tr_temp = tr_temp.float().to(device)

            # print(
            #     f"tr_data: {tr_data.shape}, tr_labels: {tr_labels.shape}, tr_features: {tr_features.shape}")

            if config["scope"] == "EmotionNet":
                tr_outputs = model(tr_data, tr_spatial, tr_temp)
            else:
                tr_outputs = model(tr_data)  # Prediction

            if config["scope"] == "EmotionNet":
                # Split the outputs into valence and arousal
                tr_outputs_valence = tr_outputs[:, 0, :].squeeze()
                tr_outputs_arousal = tr_outputs[:, 1, :].squeeze()

                # Split the labels into valence and arousal
                # tr_labels_valence = tr_labels[:, 0].long()
                # tr_labels_arousal = tr_labels[:, 1].long()

                # Compute the loss for each separately
                loss_valence = criterion(tr_outputs_valence, tr_labels_valence)
                loss_arousal = criterion(tr_outputs_arousal, tr_labels_arousal)

                # Combine the two losses
                tr_epoch_loss = loss_valence + loss_arousal
                tr_cumulative_loss += tr_epoch_loss
            else:
                # Multiclassification loss considering all classes
                tr_epoch_loss = criterion(tr_outputs, tr_labels)

            optimizer.zero_grad()
            tr_epoch_loss.backward()
            optimizer.step()

            with torch.no_grad():
                if config["scope"] == "EmotionNet":
                    tr_accuracy_valence = accuracy_metric(
                        torch.argmax(tr_outputs_valence, -1), tr_labels_valence) * 100

                    tr_accuracy_arousal = accuracy_metric(
                        torch.argmax(tr_outputs_arousal, -1), tr_labels_arousal) * 100

                    # print(
                    #     f"Arousal preds: {torch.argmax(tr_outputs_arousal, -1)}")
                    # print(f"Arousal labels: {tr_labels_arousal}")
                    # print(f"*"*50)
                    # print(
                    #     f"Valence preds: {torch.argmax(tr_outputs_valence, -1)}")
                    # print(f"Valence labels: {tr_labels_valence}")
                    # print(f"-"*50)

                    tr_accuracy = (tr_accuracy_valence +
                                   tr_accuracy_arousal) / 2

                    tr_recall_valence = recall_metric(
                        torch.argmax(tr_outputs_valence, -1), tr_labels_valence) * 100

                    tr_recall_arousal = recall_metric(
                        torch.argmax(tr_outputs_arousal, -1), tr_labels_arousal) * 100

                    tr_recall = (tr_recall_valence + tr_recall_arousal) / 2

                    epoch_tr_accuracy += tr_accuracy
                    epoch_tr_recall += tr_recall
                    tr_step += 1

                else:
                    tr_preds = torch.argmax(tr_outputs, -1).detach()
                    epoch_tr_preds = torch.cat((epoch_tr_preds, tr_preds), 0)
                    epoch_tr_labels = torch.cat(
                        (epoch_tr_labels, tr_labels), 0)

                    tr_accuracy = accuracy_metric(tr_preds, tr_labels) * 100
                    tr_recall = recall_metric(tr_preds, tr_labels) * 100
                    tr_precision = precision_metric(tr_preds, tr_labels) * 100
                    tr_f1 = f1_metric(tr_preds, tr_labels) * 100
                    tr_auroc = auroc_metric(
                        tr_outputs.softmax(dim=1), tr_labels)*100

                    if (tr_i+1) % 50 == 0:
                        print('Training -> Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%, Precision: {:.4f}%, F1: {:.4f}%, AUROC: {:.4f}%'
                              .format(epoch+1, config["epochs"], tr_i+1, total_step, tr_epoch_loss, tr_accuracy, tr_recall, tr_precision, tr_f1, tr_auroc))

        if config["use_wandb"]:
            wandb.log({"Training Loss": tr_epoch_loss.item()})
            wandb.log({"Training Accuracy": tr_accuracy.item()})
            wandb.log({"Training Recall": tr_recall.item()})
            if config["scope"] != "EmotionNet":
                wandb.log({"Training Precision": tr_precision.item()})
                wandb.log({"Training F1": tr_f1.item()})
                wandb.log({"Training AUROC": tr_auroc.item()})

        model.eval()
        with torch.no_grad():
            epoch_val_preds = torch.tensor([]).to(device)
            epoch_val_labels = torch.tensor([]).to(device)
            for _, val_batch in enumerate(val_loader):
                if config["scope"] == "AudioNet":
                    # data = audio, labels = emotions
                    val_data, val_labels = val_batch['audio'], val_batch['emotion']
                elif config["scope"] == "VideoNet":
                    # data = pixel, labels = emotions
                    val_data, val_labels = val_batch[0], val_batch[1]
                elif config["scope"] == "EmotionNet":
                    val_data, val_spatial, val_temp, val_labels_valence, val_labels_arousal = val_batch["ppg"], val_batch[
                        "ppg_spatial_features"], val_batch["ppg_temporal_features"], val_batch["valence"], val_batch["arousal"]
                else:
                    raise ValueError(f"Invalid scope: {config['scope']}")
                val_data = val_data.float().to(device)
                val_spatial = val_spatial.float().to(device)
                val_temp = val_temp.float().to(device)
                val_labels_valence = val_labels_valence.float().to(device)
                val_labels_arousal = val_labels_arousal.float().to(device)

                if config["scope"] == "EmotionNet":
                    val_outputs = model(val_data, val_spatial, val_temp)
                else:
                    val_outputs = model(val_data).to(device)

                if config["scope"] == "EmotionNet":
                    val_outputs_valence = val_outputs[:, 0, :].squeeze()
                    val_outputs_arousal = val_outputs[:, 1, :].squeeze()

                    val_labels = torch.cat(
                        (val_labels_arousal.unsqueeze(1), val_labels_valence.unsqueeze(1)), 1)

                    val_preds = torch.argmax(val_outputs, -1).detach()
                    epoch_val_preds = torch.cat(
                        (epoch_val_preds, val_preds), 0)
                    epoch_val_labels = torch.cat(
                        (epoch_val_labels, val_labels), 0)

                    val_loss_valence = criterion(
                        val_outputs_valence, val_labels_valence)
                    val_loss_arousal = criterion(
                        val_outputs_arousal, val_labels_arousal)

                    val_epoch_loss = val_loss_valence + val_loss_arousal
                    val_cumulative_loss += val_epoch_loss
                    val_step += 1

                else:
                    val_preds = torch.argmax(val_outputs, -1).detach()
                    epoch_val_preds = torch.cat(
                        (epoch_val_preds, val_preds), 0)
                    epoch_val_labels = torch.cat(
                        (epoch_val_labels, val_labels), 0)
                    # Multiclassification loss considering all classes
                    val_epoch_loss = criterion(val_outputs, val_labels)

            if config["scope"] == "EmotionNet":
                val_preds_valence = epoch_val_preds[:, 0].squeeze()
                val_preds_arousal = epoch_val_preds[:, 1].squeeze()

                epoch_val_labels_valence = epoch_val_labels[:, 0].squeeze()
                epoch_val_labels_arousal = epoch_val_labels[:, 1].squeeze()

                val_accuracy_valence = accuracy_metric(
                    val_preds_valence, epoch_val_labels_valence) * 100

                val_accuracy_arousal = accuracy_metric(
                    val_preds_arousal, epoch_val_labels_arousal) * 100

                val_accuracy = (val_accuracy_valence +
                                val_accuracy_arousal) / 2

                val_recall_valence = recall_metric(
                    val_preds_valence, epoch_val_labels_valence) * 100

                val_recall_arousal = recall_metric(
                    val_preds_arousal, epoch_val_labels_arousal) * 100

                val_recall = (val_recall_valence + val_recall_arousal) / 2

            else:
                val_accuracy = accuracy_metric(
                    epoch_val_preds, epoch_val_labels) * 100
                val_recall = recall_metric(
                    epoch_val_preds, epoch_val_labels) * 100
                val_precision = precision_metric(
                    epoch_val_preds, epoch_val_labels) * 100
                val_f1 = f1_metric(epoch_val_preds, epoch_val_labels) * 100
                val_auroc = auroc_metric(
                    val_outputs.softmax(dim=1), val_labels)*100

            if config["use_wandb"]:
                wandb.log({"Validation Loss": val_epoch_loss.item()})
                wandb.log({"Validation Accuracy": val_accuracy.item()})
                wandb.log({"Validation Recall": val_recall.item()})
                if config["scope"] != "EmotionNet":
                    wandb.log({"Validation Precision": val_precision.item()})
                    wandb.log({"Validation F1": val_f1.item()})
                    wandb.log({"Validation AUROC": val_auroc.item()})
            # print('Validation -> Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%, Precision: {:.4f}%, F1: {:.4f}%, AUROC: {:.4f}%'
            #       .format(epoch+1, config["epochs"], val_epoch_loss, val_accuracy, val_recall, val_precision, val_f1, val_auroc))

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
        epoch_tr_accuracy /= tr_step
        epoch_tr_recall /= tr_step
        tr_cumulative_loss /= tr_step

        val_cumulative_loss /= val_step

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr}')
        if config['scope'] == 'EmotionNet':
            print(
                f"Training -> Epoch [{epoch+1}/{config['epochs']}], Loss: {tr_cumulative_loss:.4f} (val: {loss_valence:.3f}, aro: {loss_arousal:.3f}), Accuracy: {epoch_tr_accuracy:.4f}% , Recall: {epoch_tr_recall:.4f}%")
            print(
                f"Validation -> Epoch [{epoch+1}/{config['epochs']}], Loss: {val_cumulative_loss:.4f}, Accuracy: {val_accuracy:.4f}%, Recall: {val_recall:.4f}%")
            print("-"*50)
