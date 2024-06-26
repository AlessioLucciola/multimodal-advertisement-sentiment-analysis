from torch import nn
from torchvision import models
from torchvision.models import DenseNet121_Weights
from config import DROPOUT_P, DATASET_NAME
import numpy as np

class VideoDenseNet121(nn.Module):
    def __init__(self, hidden_layers, num_classes, dropout_p=DROPOUT_P):
        super(VideoDenseNet121, self).__init__()
        self.model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

        if DATASET_NAME == "FER":
            self.model.features.conv0 = nn.Conv2d(
                3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # Change the number of input channels from 1 to 3

        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

        self.layers = []
        if len(hidden_layers) == 0:
            self.layers.append(self.dropout)
            self.layers.append(
                nn.Linear(self.model.classifier.in_features, num_classes, bias=False))
        else:
            self.layers.append(self.dropout)
            for i in range(len(hidden_layers)):
                if i == 0:
                    self.layers.append(
                        nn.Linear(self.model.classifier.in_features, hidden_layers[i], bias=False))
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
        self.model.classifier = self.classifier

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