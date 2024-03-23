import torch.nn as nn
from torchvision import models
from enum import Enum

# TODO: to fix -> `index 1 is out of bounds for dimension 1 with size 1`
class InceptionV3_Weights(Enum):
    DEFAULT = 'torchvision'
    IMAGENET = 'imagenet'
    PLACES365 = 'places365'

class InceptionV3(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV3, self).__init__()
        self.model = models.inception_v3(weights=InceptionV3_Weights.DEFAULT)
        out_features = 2048
        self.model.aux_logits = False
        self.model.fc = nn.Sequential(
            nn.Linear(out_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        return self.model(x)
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True