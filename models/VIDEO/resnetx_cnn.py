import torch.nn as nn
from torchvision import models
from enum import Enum

class ResNet18_Weights(Enum):
    DEFAULT = 'torchvision'
    IMAGENET = 'imagenet'
    PLACES365 = 'places365'

class ResNet34_Weights(Enum):
    DEFAULT = 'torchvision'
    IMAGENET = 'imagenet'
    PLACES365 = 'places365'

class ResNet50_Weights(Enum):
    DEFAULT = 'torchvision'
    IMAGENET = 'imagenet'
    PLACES365 = 'places365'

class ResNet101_Weights(Enum):
    DEFAULT = 'torchvision'
    IMAGENET = 'imagenet'
    PLACES365 = 'places365'


def ResNetX(model_name, num_classes):
    if model_name == 'resnet18':
        model = ResNet18(num_classes)
    elif model_name == 'resnet34':
        model = ResNet34(num_classes)
    elif model_name == 'resnet50':
        model = ResNet50(num_classes)
    elif model_name == 'resnet101':
        model = ResNet101(num_classes)
    else:
        raise ValueError('Invalid Model Name: Options [resnet18, resnet34, resnet50, resnet101]')
    
    return model

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, num_classes)

        # Add first layer to have input channels as 1
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                stride=2, padding=3, bias=False)

        self.model.fc = nn.Sequential(
            nn.Linear(512, 256),
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

class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, num_classes)

        # Add first layer to have input channels as 1
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                stride=2, padding=3, bias=False)

        self.model.fc = nn.Sequential(
            nn.Linear(512, 256),
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


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(2048, num_classes)

        # Add first layer to have input channels as 1
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                stride=2, padding=3, bias=False)

        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
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


class ResNet101(nn.Module):
    def __init__(self, num_classes):
        super(ResNet101, self).__init__()
        self.model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        self.model.fc = nn.Linear(2048, num_classes)

        # Add first layer to have input channels as 1
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                stride=2, padding=3, bias=False)

        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
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

# def get_model(num_classes, device, model_name):
#     '''This function returns the model to be used for training'''
#     if model_name == 'resnet18':
#         model = models.resnet18(pretrained=True)
#         out_features = 512
#     elif model_name == 'resnet34':
#         model = models.resnet34(pretrained=True)
#         out_features = 512
#     elif model_name == 'resnet50':
#         model = models.resnet50(pretrained=True)
#         out_features = 2048
#     elif model_name == 'resnet101':
#         model = models.resnet101(pretrained=True)
#         out_features = 2048
#     else:
#         raise ValueError('Invalid Model Name: Options [resnet18, resnet34, resnet50, resnet101]')
    
#     # add first layer to have input channels as 1
#     model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
#                             stride=2, padding=3, bias=False)

#     model.fc = nn.Sequential(
#         nn.Linear(out_features, 256),
#         nn.ReLU(),
#         nn.Dropout(0.2),
#         nn.Linear(256, num_classes),
#         nn.LogSoftmax(dim=1)
#     )
    
#     return model