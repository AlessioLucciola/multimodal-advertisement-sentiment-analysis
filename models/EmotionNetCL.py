from torch import nn
import torch
from config import DROPOUT_P, LENGTH


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Dropout(dropout)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.MaxPool2d(4)
            )

    def forward(self, x):
        out = self.layers(x)
        shortcut = self.shortcut(x)
        return nn.ReLU()(out + shortcut)


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(ConvolutionalBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)


class EmotionNet(nn.Module):
    def __init__(self, num_classes, dropout=DROPOUT_P):
        super(EmotionNet, self).__init__()
        self.num_classes = num_classes

        self.cnn_layers = nn.Sequential(
            ResidualBlock(1, 8, 3, dropout),
            ResidualBlock(8, 16, 3, dropout),
            ResidualBlock(16, 32, 3, dropout),
            nn.Flatten()
        )

        hidden_size = 32
        self.lstm = nn.LSTM(input_size=100,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        self.dropout = nn.Dropout(0.3)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, num_classes)
        )

    def forward(self, x, spatial_features):
        x_lstm = torch.squeeze(spatial_features, 1)
        x_lstm = x_lstm.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_lstm)
        lstm_out = self.dropout(lstm_out)
        last_hidden_state = lstm_out[:, -1, :]
        # spatial_features = spatial_features.unsqueeze(1)
        # cnn_out = self.cnn_layers(spatial_features)
        # out = torch.cat((cnn_out, last_hidden_state), dim=1)
        out = self.fc_layers(last_hidden_state)

        return out.view(out.size(0), self.num_classes)