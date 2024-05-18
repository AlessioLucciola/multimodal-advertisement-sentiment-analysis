import torch.nn as nn
import timm
from torchvision import transforms

from config import DROPOUT_P, DATASET_NAME

class VideoViTPretrained(nn.Module):
    def __init__(self, hidden_layers, num_classes, pretrained=True, dropout=DROPOUT_P):
        super (VideoViTPretrained, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained)

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

        if DATASET_NAME == "FER":
            self.resize = transforms.Resize((224, 224))  # Resize input to 224x224

        # Print the number of trainable parameters
        print(f'Model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} trainable params.')
        # Print the number of layers
        print(f'Model has {len(list(self.model.parameters()))} layers.')
    
    def forward(self, x):
        if DATASET_NAME == "FER":
            x = self.resize(x)  # Resize input to match model's expected size
            x = x.repeat(1, 1, 1, 1)  # Repeat the single channel input to have 1 channels
        x = self.model(x)

        return x