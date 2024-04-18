from config import DROPOUT_P, T_HEAD, T_ENC_LAYERS, T_DIM_FFW, T_KERN, T_STRIDE, T_MAXPOOL, LENGTH, WAVELET_STEP
import torch.nn as nn
import torch


class EmotionNet(nn.Module):
    def __init__(self, num_classes, dropout=DROPOUT_P):
        super().__init__()
        self.transformer_maxpool = nn.MaxPool2d(
                kernel_size=[1, T_KERN], stride=[1, T_STRIDE])

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=16, 
                out_channels=32,
                kernel_size=3,
                stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Dropout(p=dropout), 
            # nn.Conv2d(
            #     in_channels=32,
            #     out_channels=64,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1
            #           ),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=4),
            # nn.Dropout(p=dropout),
        )

        d_model = LENGTH // WAVELET_STEP
        transformer_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=T_HEAD,
                dim_feedforward=T_DIM_FFW,
                dropout=dropout,
                activation='relu',
                # batch_first=True
                )

        self.transformer_encoder = nn.TransformerEncoder(
                transformer_layer, num_layers=T_ENC_LAYERS)

        # self.fc1_linear = nn.Linear(1972, num_classes)
        self.fc1_linear = nn.Linear(d_model, num_classes)

    def forward(self, _, x):
        if T_MAXPOOL != 0:
            x = x.unsqueeze(1)
        for _ in range(T_MAXPOOL):
            x = self.transformer_maxpool(x)
            x = torch.squeeze(x, 1)

        # x_conv = x.unsqueeze(1)
        # conv_embedding = self.conv_block(x_conv)
        # conv_embedding = torch.flatten(conv_embedding, start_dim=1) 

        x = x.permute(2, 0, 1)
        transformer_output = self.transformer_encoder(x)

        # embedding = torch.cat([conv_embedding, transformer_embedding], dim=1)
        embedding = torch.mean(transformer_output, dim=0)
        out = self.fc1_linear(embedding)
        return out
