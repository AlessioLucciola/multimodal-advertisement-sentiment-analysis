import torch.nn as nn
import torchvision.transforms as transforms
import timm

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

        self.resize = transforms.Resize((224, 224))  # Resize input to 224x224

    def forward(self, x):
        x = self.resize(x)  # Resize input to match model's expected size
        x = x.repeat(1, 3, 1, 1)  # Repeat the single channel input to have 3 channels
        x = self.model(x)
        return x