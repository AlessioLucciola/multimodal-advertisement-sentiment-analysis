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
            nn.MaxPool2d(2),
            nn.Dropout(dropout)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.MaxPool2d(2)
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

        # CNN branch
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(dropout),

            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(dropout),

            nn.Flatten(),
        )
        self.cnn_layers = nn.Sequential(
            ResidualBlock(1, 16, 3, dropout),
            ResidualBlock(16, 32, 3, dropout),
            nn.Flatten()
        )

        self.cnn_layers_1d = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout),

            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=dropout),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=dropout),

            nn.Flatten()
        )
        self.lstm = nn.LSTM(input_size=20,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=False
                            )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(12128, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x, spatial_features):
        # CNN branch
        # print(f"features: {features.shape}")
        # print(spatial_features.shape)
        # print(
        #     f'x` shape: {x.shape}, spatial_features shape: {spatial_features.shape}')
        x_lstm = torch.squeeze(spatial_features, 1)
        x_lstm = x_lstm.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_lstm)
        last_hidden_state = lstm_out[:, -1, :]
        spatial_features = spatial_features.unsqueeze(1)
        cnn_out = self.cnn_layers(spatial_features)
        out = torch.cat((cnn_out, last_hidden_state), dim=1)
        # cnn_out = self.cnn_layers_1d(x.unsqueeze(1))

        # RNN branch
        # rnn_out, _ = self.rnn(temporal_features.view(
        #     temporal_features.size(0), -1, 2))

        # rnn_out = rnn_out[:, -1, :]  # Take the output from the last time step

        # # # Concatenate the outputs from the two branches
        # out = torch.cat((cnn_out, rnn_out), dim=1)
        # Fully connected layers
        # print(f"cnn_out: {cnn_out.shape}")
        out = self.fc_layers(out)

        return out.view(out.size(0), self.num_classes)
