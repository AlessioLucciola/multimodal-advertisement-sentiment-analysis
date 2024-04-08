from torch import nn
import torch
from config import DROPOUT_P


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.MaxPool1d(2)
            )

    def forward(self, x):
        out = self.layers(x)
        shortcut = self.shortcut(x)
        return nn.ReLU()(out + shortcut)


class EmotionNet(nn.Module):
    def __init__(self, num_classes, dropout=DROPOUT_P):
        super(EmotionNet, self).__init__()
        self.num_classes = num_classes

        # CNN branch
        self.cnn_layers = nn.Sequential(
            ResidualBlock(1, 10, 3, dropout),
            ResidualBlock(10, 20, 3, dropout),
            nn.Flatten(),
        )

        # RNN branch
        self.rnn = nn.RNN(1, 128, num_layers=4, batch_first=True)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(2500 + 128, 1024),  # 1500 from CNN, 64 from RNN
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2 * num_classes),
        )

    def forward(self, x, features=None):

        # CNN branch
        cnn_out = self.cnn_layers(x.view(x.size(0), 1, -1))

        # RNN branch
        rnn_out, _ = self.rnn(x.view(x.size(0), -1, 1))
        rnn_out = rnn_out[:, -1, :]  # Take the output from the last time step

        # Concatenate the outputs from the two branches
        out = torch.cat((cnn_out, rnn_out), dim=1)

        # Fully connected layers
        out = self.fc_layers(out)

        return out.view(out.size(0), 2, self.num_classes)
