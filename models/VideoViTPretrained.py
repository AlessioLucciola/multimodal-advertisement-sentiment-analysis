import torch.nn as nn
import timm
import numpy as np
from config import DROPOUT_P

class VideoViTPretrained(nn.Module):
    def __init__(self, hidden_layers, num_classes, pretrained=True, dropout=DROPOUT_P):
        super(VideoViTPretrained, self).__init__()
        self.model = timm.create_model("vit_base_patch32_224", pretrained)
        
        for param in self.model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

        self.layers = []
        if len(hidden_layers) == 0:
            self.layers.append(self.dropout)
            self.layers.append(nn.Linear(self.model.head.in_features, num_classes, bias=False))
        else:
            self.layers.append(self.dropout)
            for i in range(len(hidden_layers)):
                if i == 0:
                    self.layers.append(nn.Linear(self.model.head.in_features, hidden_layers[i], bias=False))
                    self.layers.append(self.relu)
                    self.layers.append(nn.BatchNorm1d(hidden_layers[i]))
                else:
                    self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i], bias=False))
                    self.layers.append(self.relu)
                    self.layers.append(nn.BatchNorm1d(hidden_layers[i]))
            self.layers.append(nn.Linear(hidden_layers[-1], num_classes, bias=False))
            self.layers.append(nn.BatchNorm1d(num_classes))

        self.classifier = nn.Sequential(*self.layers)
        self.model.head = self.classifier

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Model has {params} trainable params.')

    def forward(self, x):
        x = self.model(x)
        return x