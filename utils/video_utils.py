from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
from torchvision import transforms
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from shared.constants import FER_emotion_mapping

# Show images
def show_images(dataloader, title='Images'):
    '''This function plots the images from the dataloader'''
    fig, ax = plt.figure(figsize=(16, 8)), plt.axis("off")
    for images, _ in dataloader:
        print('Images Shape:', images.shape)
        plt.imshow(make_grid(images, nrow=8).permute(
            (1, 2, 0)))  # move the channel dimension
        break

    plt.suptitle(f"{title}", y=0.92, fontsize=12)
    plt.show()