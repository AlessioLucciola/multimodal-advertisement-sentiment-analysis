
from config import DROPOUT_P, EMOTION_NUM_CLASSES, WT
import torch.nn as nn

class EmotionNet(nn.Module):
    def __init__(self, dropout_p=DROPOUT_P):
        super().__init__() 
        self.CNN_block_1d = nn.Sequential(
            nn.Conv1d(
                in_channels=3,
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
            nn.Flatten()
        )
         

        self.fc = nn.Sequential(
                nn.Linear(1536, 1024),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(256, EMOTION_NUM_CLASSES))

    def forward(self, x):
        final= self.CNN_block_1d(x)
        out = self.fc(final)  
        return out
