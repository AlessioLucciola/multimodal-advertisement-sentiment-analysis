from config import BATCH_SIZE
from pyteap.signals.bvp import get_bvp_features
import numpy as np
import torch
from torch import nn


class EmotionNet(nn.Module):
    def __init__(self, num_classes):
        super(EmotionNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)  # Add batch normalization
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)  # Add batch normalization
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 500, 2 * num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1)
        out = self.conv1(x)
        out = self.bn1(out)  # Apply batch normalization
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.bn2(out)  # Apply batch normalization
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out.view(-1, 2, self.num_classes)
