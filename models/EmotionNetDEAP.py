
from config import DROPOUT_P, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, EMOTION_NUM_CLASSES, WT, LENGTH
import torch.nn as nn
import torch
import torch.nn.functional as F

class EmotionNet(nn.Module):
    def __init__(self, num_classes, input_size, num_layers=LSTM_NUM_LAYERS, hidden_size=LSTM_HIDDEN_SIZE, dropout_p=DROPOUT_P):
        super().__init__() 
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            # dropout=dropout_p,
                            bidirectional=False)
        self.CNN_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(
                in_channels=64, 
                out_channels=128,
                kernel_size=3,
                stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=dropout_p), 
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=dropout_p),
            nn.Flatten()
        )

        self.CNN_block_1d = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=32,
                kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout_p),
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout_p),
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout_p),
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout_p),
            nn.Flatten()
        )
         

        self.fc = nn.Sequential(
                nn.Linear(1536, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, EMOTION_NUM_CLASSES))

        self.fc2 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, EMOTION_NUM_CLASSES))

    def forward(self, x):
        if not WT:
            final= self.CNN_block_1d(x)
            out = self.fc(final)  
            return out
        else:
            final= self.CNN_block_1d(x)
            out = self.fc(final)  
            return out
            x = x.squeeze()
            x = x.permute(0,2,1)
            final, _ = self.lstm(x)
            last = final[:,-1,:] 
            out = self.fc2(last)
            return out
