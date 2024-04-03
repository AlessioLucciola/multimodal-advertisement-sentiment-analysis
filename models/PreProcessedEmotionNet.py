from config import BATCH_SIZE
from pyteap.signals.bvp import get_bvp_features
import numpy as np
import torch
from torch import nn
from utils.utils import select_device


class PreProcessedEmotionNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(PreProcessedEmotionNet, self).__init__()
        self.device = select_device()
        self.input_size = input_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes * 2)
        self.fc = nn.Linear(input_size, num_classes * 2)

    def forward(self, x):
        # out = self.fc1(x)
        # out = self.relu(out)
        # out = self.fc2(out)
        # out = out.view(-1, 2, self.num_classes)
        return self.fc(x).view(-1, 2, self.num_classes)
