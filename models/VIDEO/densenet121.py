import torch.nn as nn
from torchvision import models
from enum import Enum
from torchvision.models import DenseNet121_Weights

class DenseNet121(nn.Module):
    def __init__(self, num_classes, dropout_p):
        super(DenseNet121, self).__init__()
        self.model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        out_features = 1024
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.classifier = nn.Sequential(
            nn.Linear(out_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)   
        )
        
    def forward(self, x):
        return self.model(x)
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True