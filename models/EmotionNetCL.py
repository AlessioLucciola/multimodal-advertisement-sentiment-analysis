from config import DROPOUT_P
import torch.nn as nn
import torch

class EmotionNet(nn.Module):
    def __init__(self, num_classes, dropout=DROPOUT_P):
        super().__init__() 
        self.conv_block_xs = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Dropout(p=0.1),
            nn.Conv2d(
                in_channels=16, 
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Dropout(p=0.2)
        )


        self.conv_block_x = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=8,
                kernel_size=8,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # nn.Dropout(p=dropout),
        )
        self.fc1_linear_xs = nn.Linear(4096, num_classes) 
        self.fc1_linear_x = nn.Sequential(
                nn.Linear(1576, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes))
        self.fc1_linear_combo = nn.Sequential(
                nn.Linear(1960, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes))

        # self.fc1_linear_x = nn.Linear(1600, num_classes) 
    def forward(self, x, x_s):
        x_s = x_s.unsqueeze(1)
        # print(f"x shape is: {x.shape}")
        x_s = self.conv_block_xs(x_s)
        x_s =  torch.flatten(x_s, start_dim=1) 

        x = x.unsqueeze(1) 
        x = self.conv_block_x(x)
        x = torch.flatten(x, start_dim=1) 


        lin_in = torch.cat((x, x_s), dim=1)
        out = self.fc1_linear_combo(lin_in)  
        return out

