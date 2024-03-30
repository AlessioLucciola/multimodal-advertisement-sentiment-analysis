import torch.nn as nn
from torchvision import models
from enum import Enum
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights


def VideoResNetX(model_name, num_classes, dropout_p):
    if model_name == 'resnet18':
        model = ResNet18(num_classes, dropout_p)
    elif model_name == 'resnet34':
        model = ResNet34(num_classes, dropout_p)
    elif model_name == 'resnet50':
        model = ResNet50(num_classes, dropout_p)
    elif model_name == 'resnet101':
        model = ResNet101(num_classes, dropout_p)
    else:
        raise ValueError('Invalid Model Name: Options [resnet18, resnet34, resnet50, resnet101]')
    
    return model

class ResNet18(nn.Module):
    def __init__(self, num_classes, dropout_p):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, num_classes)

        # Add first layer to have input channels as 1
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                stride=2, padding=3, bias=False)

        self.model.fc = nn.Sequential(
            nn.Linear(512, 256),
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
        for param in self.model.fc.parameters():
            param.requires_grad = True

class ResNet34(nn.Module):
    def __init__(self, num_classes, dropout_p):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, num_classes)

        # Add first layer to have input channels as 1
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                stride=2, padding=3, bias=False)

        self.model.fc = nn.Sequential(
            nn.Linear(512, 256),
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
        for param in self.model.fc.parameters():
            param.requires_grad = True


class ResNet50(nn.Module):
    def __init__(self, num_classes, dropout_p):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(2048, num_classes)

        # Add first layer to have input channels as 1
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                stride=2, padding=3, bias=False)

        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
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
        for param in self.model.fc.parameters():
            param.requires_grad = True


class ResNet101(nn.Module):
    def __init__(self, num_classes, dropout_p):
        super(ResNet101, self).__init__()
        self.model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        self.model.fc = nn.Linear(2048, num_classes)

        # Add first layer to have input channels as 1
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                stride=2, padding=3, bias=False)

        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
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
        for param in self.model.fc.parameters():
            param.requires_grad = True