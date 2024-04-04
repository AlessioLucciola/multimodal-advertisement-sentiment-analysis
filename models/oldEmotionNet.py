import torch.nn as nn
import torch.nn.functional as F


class EmotionNet(nn.Module):
    def __init__(self, num_classes, dropout):
        super(EmotionNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(1, 512, 50)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(512, 256, 25)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.lstm1 = nn.LSTM(475, 256)
        self.lstm2 = nn.LSTM(256, 128)
        self.fc1 = nn.Linear(32768, 256)
        self.dropout3 = nn.Dropout(0.5)
        # Adjusted to output a tensor of shape (batch_size, 10)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes * 2)

    def forward(self, x, features):
        # print(f"Input shape: {x.shape}")
        x = x.view(x.size(0), 1, -1)
        x = F.relu(self.conv1(x))
        # print(f"Conv1 shape: {x.shape}")
        x = F.max_pool1d(x, 2)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = self.bn2(x)
        x = self.dropout2(x)
        # print(f"Conv2 shape: {x.shape}")
        x, _ = self.lstm1(x)
        # print(f"LSTM1 shape: {x.shape}")
        x, _ = self.lstm2(x)
        x = x.view(x.size(0), -1)  # Flatten
        # print(f"LSTM2 shape: {x.shape}")
        x = F.sigmoid(self.fc1(x))
        x = self.dropout3(x)
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        # Reshape to (batch_size, 2, 5)
        x = x.view(x.size(0), 2, self.num_classes)
        return x
