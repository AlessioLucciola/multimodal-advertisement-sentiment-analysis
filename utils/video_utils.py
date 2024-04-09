import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from models.VideoResnetX import VideoResNetX
from models.VideoDenseNet121 import VideoDenseNet121
from models.VideoCustomCNN import VideoCustomCNN
from models.VideoViTPretrained import VideoViTPretrained

# Plots the images from the dataloader
def show_images(dataloader, title='Images'):
    fig, ax = plt.figure(figsize=(16, 8)), plt.axis("off")
    for images, _ in dataloader:
        print('Images Shape:', images.shape)
        plt.imshow(make_grid(images, nrow=8).permute(
            (1, 2, 0)))  # move the channel dimension
        break

    plt.suptitle(f"{title}", y=0.92, fontsize=12)
    plt.show()

# Select model
def select_model(model_name, hidden_size, num_classes, dropout_p):
    if model_name == 'resnet18' or model_name == 'resnet34' or model_name == 'resnet50' or model_name == 'resnet101':
        model = VideoResNetX(model_name, hidden_size, num_classes, dropout_p)
    elif model_name == 'densenet121':
        model = VideoDenseNet121(hidden_size, num_classes, dropout_p)
    elif model_name == 'custom-cnn':
        model = VideoCustomCNN(num_classes, dropout_p)
    elif model_name == 'vit-pretrained':
        model = VideoViTPretrained(hidden_size, num_classes, dropout_p)
    else:
        raise ValueError('Invalid Model Name: Options [resnet18, resnet34, resnet50, resnet101, densenet121, custom-cnn, vit-pretrained]')
    
    return model