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
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        # Adjust the dimensions in the fully connected layer
        # 2 outputs * num_classes
        self.fc1 = nn.Linear(128 * 500, 2 * num_classes)

    def forward(self, x):
        # Reshape to (batch_size, 1, sequence_length)
        x = x.view(x.shape[0], 1, -1)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)  # Flatten the tensor
        out = self.fc1(out)
        # reshape to (batch_size, 2, num_classes)
        return out.view(-1, 2, self.num_classes)

    # def forward(self, x):
    #     out = torch.tensor([])
    #     for i in range(x.shape[0]):
    #         try:
    #             result = torch.tensor(get_bvp_features(
    #                 x[i].numpy(), sr=90)).to(torch.float32)
    #         except Exception as e:
    #             result = torch.zeros(17).to(torch.float32)
    #         # print(f"result shape is {result.shape}")
    #         out = torch.cat((out, result.unsqueeze(0)), 0)
    #     # out = self.fc1(out)
    #     # out = self.relu(out)
    #     out = self.fc1(out)
    #     # reshape to (batch_size, 2, num_classes)
    #     return out.view(-1, 2, self.num_classes)
