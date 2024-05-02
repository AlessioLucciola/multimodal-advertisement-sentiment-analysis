import torch.nn as nn
from torchvision import models
import numpy as np
from config import DROPOUT_P, VIDEO_DATASET_NAME
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights

def VideoResNetX(model_name, hidden_size, num_classes, dropout_p=DROPOUT_P):
    if model_name == 'resnet18':
        model = ResNet18(hidden_size, num_classes, dropout_p)
    elif model_name == 'resnet34':
        model = ResNet34(hidden_size, num_classes, dropout_p)
    elif model_name == 'resnet50':
        model = ResNet50(hidden_size, num_classes, dropout_p)
    elif model_name == 'resnet101':
        model = ResNet101(hidden_size, num_classes, dropout_p)
    else:
        raise ValueError('Invalid Model Name: Options [resnet18, resnet34, resnet50, resnet101]')
    
    return model

def if_dataset_name_is_fer(model):
    if VIDEO_DATASET_NAME == "fer":
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Change the number of input channels from 1 to 3
    return model

class ResNet18(nn.Module):
    def __init__(self, hidden_layers, num_classes, dropout_p=DROPOUT_P):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        self.model = if_dataset_name_is_fer(self.model)

        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()
        # In features: 512
        self.layers = []
        if len(hidden_layers) == 0:
            self.layers.append(self.dropout)
            self.layers.append(
                nn.Linear(self.model.fc.in_features, num_classes, bias=False))
        else:
            self.layers.append(self.dropout)
            for i in range(len(hidden_layers)):
                if i == 0:
                    self.layers.append(
                        nn.Linear(self.model.fc.in_features, hidden_layers[i], bias=False))
                    self.layers.append(self.relu)
                    self.layers.append(nn.BatchNorm1d(hidden_layers[i]))
                else:
                    self.layers.append(
                        nn.Linear(hidden_layers[i-1], hidden_layers[i], bias=False))
                    self.layers.append(self.relu)
                    self.layers.append(nn.BatchNorm1d(hidden_layers[i]))
            self.layers.append(
                nn.Linear(hidden_layers[-1], num_classes, bias=False))
            self.layers.append(nn.BatchNorm1d(num_classes))

        self.classifier = nn.Sequential(*self.layers)
        self.model.fc = self.classifier

        model_parameters = filter(
            lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Model has {params} trainable params.')

    def forward(self, x):
        return self.model(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ResNet34(nn.Module):
    def __init__(self, hidden_layers, num_classes, dropout_p=DROPOUT_P):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(weights=ResNet34_Weights.DEFAULT)

        self.model = if_dataset_name_is_fer(self.model)

        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()
        # In features: 512
        self.layers = []
        if len(hidden_layers) == 0:
            self.layers.append(self.dropout)
            self.layers.append(
                nn.Linear(self.model.fc.in_features, num_classes, bias=False))
        else:
            self.layers.append(self.dropout)
            for i in range(len(hidden_layers)):
                if i == 0:
                    self.layers.append(
                        nn.Linear(self.model.fc.in_features, hidden_layers[i], bias=False))
                    self.layers.append(self.relu)
                    self.layers.append(nn.BatchNorm1d(hidden_layers[i]))
                else:
                    self.layers.append(
                        nn.Linear(hidden_layers[i-1], hidden_layers[i], bias=False))
                    self.layers.append(self.relu)
                    self.layers.append(nn.BatchNorm1d(hidden_layers[i]))
            self.layers.append(
                nn.Linear(hidden_layers[-1], num_classes, bias=False))
            self.layers.append(nn.BatchNorm1d(num_classes))

        self.classifier = nn.Sequential(*self.layers)
        self.model.fc = self.classifier

        model_parameters = filter(
            lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Model has {params} trainable params.')

    def forward(self, x):
        return self.model(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ResNet50(nn.Module):
    def __init__(self, hidden_layers, num_classes, dropout_p=DROPOUT_P):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        self.model = if_dataset_name_is_fer(self.model)

        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()
        # In features: 512
        self.layers = []
        if len(hidden_layers) == 0:
            self.layers.append(self.dropout)
            self.layers.append(
                nn.Linear(self.model.fc.in_features, num_classes, bias=False))
        else:
            self.layers.append(self.dropout)
            for i in range(len(hidden_layers)):
                if i == 0:
                    self.layers.append(
                        nn.Linear(self.model.fc.in_features, hidden_layers[i], bias=False))
                    self.layers.append(self.relu)
                    self.layers.append(nn.BatchNorm1d(hidden_layers[i]))
                else:
                    self.layers.append(
                        nn.Linear(hidden_layers[i-1], hidden_layers[i], bias=False))
                    self.layers.append(self.relu)
                    self.layers.append(nn.BatchNorm1d(hidden_layers[i]))
            self.layers.append(
                nn.Linear(hidden_layers[-1], num_classes, bias=False))
            self.layers.append(nn.BatchNorm1d(num_classes))

        self.classifier = nn.Sequential(*self.layers)
        self.model.fc = self.classifier

        model_parameters = filter(
            lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Model has {params} trainable params.')

    def forward(self, x):
        return self.model(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ResNet101(nn.Module):
    def __init__(self, hidden_layers, num_classes, dropout_p=DROPOUT_P):
        super(ResNet101, self).__init__()
        self.model = models.resnet101(weights=ResNet101_Weights.DEFAULT)

        self.model = if_dataset_name_is_fer(self.model)

        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()
        # In features: 512
        self.layers = []
        if len(hidden_layers) == 0:
            self.layers.append(self.dropout)
            self.layers.append(
                nn.Linear(self.model.fc.in_features, num_classes, bias=False))
        else:
            self.layers.append(self.dropout)
            for i in range(len(hidden_layers)):
                if i == 0:
                    self.layers.append(
                        nn.Linear(self.model.fc.in_features, hidden_layers[i], bias=False))
                    self.layers.append(self.relu)
                    self.layers.append(nn.BatchNorm1d(hidden_layers[i]))
                else:
                    self.layers.append(
                        nn.Linear(hidden_layers[i-1], hidden_layers[i], bias=False))
                    self.layers.append(self.relu)
                    self.layers.append(nn.BatchNorm1d(hidden_layers[i]))
            self.layers.append(
                nn.Linear(hidden_layers[-1], num_classes, bias=False))
            self.layers.append(nn.BatchNorm1d(num_classes))

        self.classifier = nn.Sequential(*self.layers)
        self.model.fc = self.classifier

        model_parameters = filter(
            lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Model has {params} trainable params.')

    def forward(self, x):
        return self.model(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)