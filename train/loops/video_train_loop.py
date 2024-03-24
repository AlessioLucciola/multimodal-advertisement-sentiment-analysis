from tqdm import tqdm
import torch
import wandb
from datetime import datetime
import copy
from config import SAVE_MODELS, SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH, N_EPOCHS
from sklearn.metrics import recall_score
from utils.utils import save_results, save_model, save_configurations
from utils.video_utils import evaluate_model, get_accuracy

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
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(N_EPOCHS):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        curr_len = 0
        batch_id = 0
        for images, labels in tqdm(train_loader, ascii=True, desc=f"Epoch: {epoch+1:>{len(str(N_EPOCHS))}}/{N_EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = get_accuracy(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * images.shape[0]
            running_acc += acc
            curr_len += images.shape[0]
            batch_id += 1


            curr_loss = running_loss/curr_len
            curr_acc = running_acc/batch_id
            cur_recall = recall_score(labels.cpu().numpy(), torch.argmax(outputs, -1).cpu().numpy(), average='macro')

            if (epoch+1) % 5 == 0:
                    print('Training -> Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%'
                            .format(epoch+1, config["epochs"], epoch+1, total_step, curr_loss, curr_acc, cur_recall))
        
        if config["use_wandb"]:
            wandb.log({"Training Loss": curr_loss.item()})
            wandb.log({"Training Accuracy": curr_acc})
            wandb.log({"Training Recall": cur_recall})

        running_loss /= len(train_loader.dataset)
        running_acc /= len(train_loader)
        train_losses.append(running_loss)
        train_accuracies.append(running_acc)
        val_loss, val_acc, cm = evaluate_model(
            model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_recall = 0 # TODO: TO BE IMPLEMENTED

        if config["use_wandb"]:
                wandb.log({"Validation Loss": val_loss.item()})
                wandb.log({"Validation Accuracy": val_acc})
                wandb.log({"Validation Recall": val_recall})
        print('Validation -> Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%'
                .format(epoch+1, config["epochs"], val_loss, val_acc, val_recall))

        if best_accuracy is None or val_acc < best_accuracy:
                best_accuracy = val_acc
                best_model = copy.deepcopy(model)
        current_results = {
            'epoch': epoch+1,
            'validation_loss': val_loss.item(),
            'training_loss': curr_loss.item(),
            'validation_accuracy': val_acc,
            'training_accuracy': curr_acc,
            'validation_recall': val_recall,
            'training_recall': cur_recall
        }
        if SAVE_RESULTS:
            save_results(data_name, current_results)
        if SAVE_MODELS:
            save_model(data_name, model, epoch)
        if epoch == config["epochs"]-1 and SAVE_MODELS:
            save_model(data_name, best_model, epoch=None, is_best=True)

# def train_eval_loop(model, train_loader, val_loader, device, params):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(
#         model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=params['lr'],
#                                                     epochs=params['num_epochs'],
#                                                     steps_per_epoch=len(train_loader))

#     train_losses = []
#     train_accuracies = []
#     val_losses = []
#     val_accuracies = []
#     best_loss = np.inf

#     for epoch in range(params['num_epochs']):
#         print(f"|--------- Epoch: {epoch+1:>{len(str(params['num_epochs']))}}/{params['num_epochs']} " + "-"*110)
#         model.train()
#         running_loss = 0.0
#         running_acc = 0.0
#         curr_len = 0
#         batch_id = 0
#         for images, labels in tqdm(train_loader, ascii=True, desc=f"Epoch: {epoch+1:>{len(str(params['num_epochs']))}}/{params['num_epochs']}"):
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             acc = get_accuracy(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#             running_loss += loss.item() * images.shape[0]
#             running_acc += acc
#             curr_len += images.shape[0]
#             batch_id += 1


#             curr_loss = running_loss/curr_len
#             curr_acc = running_acc/batch_id

#             if batch_id % 10 == 0:
#                 print('\t\t'+'-'*70)
#                 print(
#                     f"\t\t| Batch: {batch_id:>{len(str(len(train_loader)))}}/{len(train_loader)} | Training Loss: {curr_loss:.4f} | Training Accuracy: {curr_acc:.4f} |")
#                 print('\t\t'+'-'*70)

#         running_loss /= len(train_loader.dataset)
#         running_acc /= len(train_loader)
#         train_losses.append(running_loss)
#         train_accuracies.append(running_acc)
#         val_loss, val_acc, cm = evaluate_model(
#             model, val_loader, criterion, device)
#         val_losses.append(val_loss)
#         val_accuracies.append(val_acc)

#         # Save the model if the validation loss has decreased
#         if val_loss < best_loss:
#             best_loss = val_loss
#             best_model_state = copy.deepcopy(model.state_dict())
#             model.load_state_dict(best_model_state)
#             torch.save(model.state_dict(), 'checkpoints/video/' + params['model_name'] + '_best.pt')

#         # Save the model every 5 epochs and plot the results
#         if SAVE_MODELS and (epoch != 0 or epoch % 5 == 0):
#             torch.save(model.state_dict(), 'checkpoints/video/' + params['model_name'] + '_' + str(epoch+1) + '.pt')
#             plot_results((train_losses, train_accuracies, val_losses, val_accuracies), params['model_name'])

#         print('-'*120)
#         print(f"Epoch: {epoch+1:>{len(str(params['num_epochs']))}}/{params['num_epochs']} | Training Loss: {running_loss:.4f} | Training Accuracy: {running_acc:.4f} | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")
#         print('-'*120)
#         print("-"*130 + '-|')

#     return model, (train_losses, train_accuracies, val_losses, val_accuracies)