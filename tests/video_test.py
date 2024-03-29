from utils.utils import save_results, set_seed, select_device, upload_scaler
from config import RANDOM_SEED, SAVE_RESULTS, PATH_TO_SAVE_RESULTS, VAL_SIZE, LIMIT, MODEL_NAME, BATCH_SIZE, METADATA_CSV, NUM_CLASSES, DROPOUT_P
from torchmetrics import Accuracy, Recall, Precision, F1Score, AUROC
from dataloaders.FER_dataloader import FERDataloader
from models.AudioNetCT import AudioNet_CNN_Transformers as AudioNetCT
from models.AudioNetCL import AudioNet_CNN_LSTM as AudioNetCL
from tqdm import tqdm
import torch
import os
import json

from models.VIDEO.densenet121 import DenseNet121
from models.VIDEO.inception_v3 import InceptionV3
from models.VIDEO.resnetx_cnn import ResNetX
from models.VIDEO.custom_cnn import CustomCNN

def test_loop(test_model, test_loader, device, model_path, criterion):
    test_model.eval()
    test_loss_iter = 0
    accuracy_metric = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)
    recall_metric = Recall(task="multiclass", num_classes=NUM_CLASSES, average='macro').to(device)
    precision_metric = Precision(task="multiclass", num_classes=NUM_CLASSES, average='macro').to(device)
    f1_metric = F1Score(task="multiclass", num_classes=NUM_CLASSES, average='macro').to(device)
    auroc_metric = AUROC(task="multiclass", num_classes=NUM_CLASSES).to(device)

    with torch.no_grad():
        epoch_test_preds = torch.tensor([]).to(device)
        epoch_test_labels = torch.tensor([], dtype=torch.long).to(device)
        epoch_test_probs = torch.tensor([]).to(device)
        for _, tr_batch in enumerate(tqdm(test_loader, desc="Testing model..", leave=False)):
            type = model_path.split('_')[0]
            if type == "AudioNet":
                test_data, test_labels = tr_batch['audio'], tr_batch['emotion'] # data = audio, labels = emotions
            if type == "Video_Emotion_Recognition":
                    test_data, test_labels = tr_batch[0], tr_batch[1] # data = pixel, labels = emotions
            test_data = test_data.to(device)
            test_labels = test_labels.to(device)

            test_outputs = test_model(test_data)
            test_preds = torch.argmax(test_outputs, -1).detach()
            test_probs = test_outputs.softmax(dim=1)
            epoch_test_preds = torch.cat((epoch_test_preds, test_preds), 0)
            epoch_test_labels = torch.cat((epoch_test_labels, test_labels), 0)
            epoch_test_probs = torch.cat((epoch_test_probs, test_probs), 0)

            # Multiclassification loss considering all classes
            test_epoch_loss = criterion(test_outputs, test_labels)
            test_loss_iter += test_epoch_loss.item()

        test_loss = test_loss_iter / (len(test_loader) * test_loader.batch_size)
        test_accuracy = accuracy_metric(epoch_test_preds, epoch_test_labels) * 100
        test_recall = recall_metric(epoch_test_preds, epoch_test_labels) * 100
        test_precision = precision_metric(epoch_test_preds, epoch_test_labels) * 100
        test_f1 = f1_metric(epoch_test_preds, epoch_test_labels) * 100
        test_auroc = auroc_metric(epoch_test_probs, epoch_test_labels) * 100

        print('Test -> Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%, Precision: {:.4f}%, F1: {:.4f}%, AUROC: {:.4f}%'.format(
            test_loss, test_accuracy, test_recall, test_precision, test_f1, test_auroc))

        test_results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy.item(),
            'test_recall': test_recall.item(),
            'test_precision': test_precision.item(),
            'test_f1': test_f1.item(),
            'test_auroc': test_auroc.item()
        }
        if SAVE_RESULTS:
            save_results(model_path, test_results, test=True)

def get_model_and_dataloader(model_path, device):
    # Load configuration
    conf_path = PATH_TO_SAVE_RESULTS + f"/{model_path}/configurations.json"
    configurations = None
    if os.path.exists(conf_path):
        print(
            "--Model-- Old configurations found. Using those configurations for the test.")
        with open(conf_path, 'r') as json_file:
            configurations = json.load(json_file)
    else:
        print("--Model-- Old configurations NOT found. Using configurations in the config for test.")

    type = model_path.split('_')[0]
    model = None
    dataloader = None
    if type == "Video_Emotion_Recognition":
        # load the model
        if MODEL_NAME == 'resnet18' or MODEL_NAME == 'resnet34' or MODEL_NAME == 'resnet50' or MODEL_NAME == 'resnet101':
            model = ResNetX(MODEL_NAME, NUM_CLASSES, DROPOUT_P)
        elif MODEL_NAME == 'dense121':
            model = DenseNet121(NUM_CLASSES, DROPOUT_P)
        elif MODEL_NAME == 'inception_v3':
            model = InceptionV3(NUM_CLASSES, DROPOUT_P)
        elif MODEL_NAME == 'custom_cnn':
            model = CustomCNN(NUM_CLASSES, DROPOUT_P)
        else:
            raise ValueError('Invalid Model Name: Options [resnet18, resnet34, resnet50, resnet101, dense121, inception_v3, custom_cnn]')

        model = model.to(device)
        dataloader = FERDataloader(csv_file=METADATA_CSV,
                                   batch_size=BATCH_SIZE,
                                   val_size=VAL_SIZE,
                                   seed=RANDOM_SEED,
                                   limit=LIMIT)
    else:
        raise ValueError(f"Unknown architecture {type}")

    return model, dataloader

def load_test_model(model, model_path, epoch, device):
    state_dict = torch.load(
        f"{PATH_TO_SAVE_RESULTS}/{model_path}/models/mi_project_{epoch}.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main(model_path, epoch):
    set_seed(RANDOM_SEED)
    device = select_device()
    model, dataloader = get_model_and_dataloader(model_path, device)
    model = load_test_model(model, model_path, epoch, device)
    test_loader = dataloader.get_test_dataloader()
    criterion = torch.nn.CrossEntropyLoss()
    test_loop(model, test_loader, device, model_path, criterion)


if __name__ == "__main__":
    # Name of the sub-folder into "results" folder in which to find the model to test (e.g. "resnet34_2023-12-10_12-29-49")
    model_path = "mi_project"
    # Specify the epoch number (e.g. 2) or "best" to get best model
    epoch = "1"

    main(model_path, epoch)

# import torch
# from config import RANDOM_SEED, USE_WANDB, VAL_SIZE, LIMIT, MODEL_NAME, BATCH_SIZE, LR, N_EPOCHS, METADATA_CSV, REG, NUM_CLASSES, DROPOUT_P
# from dataloaders.FER_dataloader import FERDataloader
# from utils.utils import set_seed, select_device
# from utils.video_utils import evaluate_model, plot_confusion_matrix, plot_roc

# from models.VIDEO.densenet121 import DenseNet121
# from models.VIDEO.inception_v3 import InceptionV3
# from models.VIDEO.resnetx_cnn import ResNetX
# from models.VIDEO.custom_cnn import CustomCNN

# def load_trained_model(model_path):
#     device = select_device()


#     # load the model
#     if MODEL_NAME == 'resnet18' or MODEL_NAME == 'resnet34' or MODEL_NAME == 'resnet50' or MODEL_NAME == 'resnet101':
#         model = ResNetX(MODEL_NAME, NUM_CLASSES, DROPOUT_P)
#     elif MODEL_NAME == 'dense121':
#         model = DenseNet121(NUM_CLASSES, DROPOUT_P)
#     elif MODEL_NAME == 'inception_v3':
#         model = InceptionV3(NUM_CLASSES, DROPOUT_P)
#     elif MODEL_NAME == 'custom_cnn':
#         model = CustomCNN(NUM_CLASSES, DROPOUT_P)
#     else:
#         raise ValueError('Invalid Model Name: Options [resnet18, resnet34, resnet50, resnet101, dense121, inception_v3, custom_cnn]')

#     model = model.to(device)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()

#     return model


# def main():
#     set_seed(RANDOM_SEED)
#     device = select_device()
#     fer_dataloader = FERDataloader(csv_file=METADATA_CSV,
#                                    batch_size=BATCH_SIZE,
#                                    val_size=VAL_SIZE,
#                                    seed=RANDOM_SEED,
#                                    limit=LIMIT)
    
#     test_loader = fer_dataloader.get_test_dataloader()
    
#     # Load the trained model
#     model = load_trained_model('./checkpoints/video/'+ MODEL_NAME + '_best.pt') # change this to the path of the trained model
#     criterion = torch.nn.CrossEntropyLoss()
    
#     test_loss, test_acc, test_cm  = evaluate_model(
#         model, test_loader, criterion, device)

#     print("-"*100)
#     print(f"| Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f} |")
#     print("-"*100)

#     # Plot the results
#     plot_roc(model, test_loader, device, model_name=MODEL_NAME)
#     plot_confusion_matrix(test_cm, model_name=MODEL_NAME)

# if __name__ == "__main__":
#     main()