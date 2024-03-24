import torch
from config import RANDOM_SEED, USE_WANDB, VAL_SIZE, LIMIT, MODEL_NAME, BATCH_SIZE, LR, N_EPOCHS, METADATA_CSV, REG, NUM_CLASSES 
from dataloaders.FER_dataloader import FERDataloader
from utils.utils import set_seed, select_device
from utils.video_utils import evaluate_model, plot_confusion_matrix, plot_roc

from models.VIDEO.densenet121 import DenseNet121
from models.VIDEO.inception_v3 import InceptionV3
from models.VIDEO.resnetx_cnn import ResNetX
from models.VIDEO.custom_cnn import CustomCNN

def load_trained_model(model_path):
    device = select_device()


    # load the model
    if MODEL_NAME == 'resnet18' or MODEL_NAME == 'resnet34' or MODEL_NAME == 'resnet50' or MODEL_NAME == 'resnet101':
        model = ResNetX(MODEL_NAME, NUM_CLASSES)
    elif MODEL_NAME == 'dense121':
        model = DenseNet121(NUM_CLASSES)
    elif MODEL_NAME == 'inception_v3':
        model = InceptionV3(NUM_CLASSES)
    elif MODEL_NAME == 'custom_cnn':
        model = CustomCNN(NUM_CLASSES)
    else:
        raise ValueError('Invalid Model Name: Options [resnet18, resnet34, resnet50, resnet101, dense121, inception_v3, custom_cnn]')

    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def main():
    set_seed(RANDOM_SEED)
    device = select_device()
    fer_dataloader = FERDataloader(csv_file=METADATA_CSV,
                                   batch_size=BATCH_SIZE,
                                   val_size=VAL_SIZE,
                                   seed=RANDOM_SEED,
                                   limit=LIMIT)
    
    test_loader = fer_dataloader.get_test_dataloader()
    
    if MODEL_NAME == 'resnet18' or MODEL_NAME == 'resnet34' or MODEL_NAME == 'resnet50' or MODEL_NAME == 'resnet101':
        model = ResNetX(MODEL_NAME, NUM_CLASSES)
    elif MODEL_NAME == 'dense121':
        model = DenseNet121(NUM_CLASSES)
    elif MODEL_NAME == 'inception_v3':
        model = InceptionV3(NUM_CLASSES)
    elif MODEL_NAME == 'custom_cnn':
        model = CustomCNN(NUM_CLASSES)
    else:
        raise ValueError('Invalid Model Name: Options [resnet18, resnet34, resnet50, resnet101, dense121, inception_v3, custom_cnn]')
    
    # Load the trained model
    model = load_trained_model('./checkpoints/video/'+ MODEL_NAME + '_best.pt') # change this to the path of the trained model
    criterion = torch.nn.CrossEntropyLoss()
    
    test_loss, test_acc, test_cm  = evaluate_model(
        model, test_loader, criterion, device)

    print("-"*100)
    print(f"| Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f} |")
    print("-"*100)

    # Plot the results
    plot_roc(model, test_loader, device, model_name=MODEL_NAME)
    plot_confusion_matrix(test_cm, model_name=MODEL_NAME)

if __name__ == "__main__":
    main()