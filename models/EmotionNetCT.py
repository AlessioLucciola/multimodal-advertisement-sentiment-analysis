from config import DROPOUT_P, T_HEAD, T_ENC_LAYERS, T_DIM_FFW, T_KERN, T_STRIDE, T_MAXPOOL
import torch.nn as nn
import torch


class EmotionNet(nn.Module):
    def __init__(self, num_classes, dropout=DROPOUT_P):
        super().__init__()
        self.transformer_maxpool = nn.MaxPool2d(
                kernel_size=[1, T_KERN], stride=[1, T_STRIDE])

        transformer_layer = nn.TransformerEncoderLayer(
                d_model=100,
                nhead=T_HEAD,
                dim_feedforward=T_DIM_FFW,
                dropout=dropout,
                activation='relu',
                batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(
                transformer_layer, num_layers=T_ENC_LAYERS)

        self.fc1_linear = nn.Linear(100, num_classes)

    def forward(self, _, x):
        if T_MAXPOOL != 0:
            x = x.unsqueeze(1)
        for _ in range(T_MAXPOOL):
            x = self.transformer_maxpool(x)
            x = torch.squeeze(x, 1)
        x = x.permute(2, 0, 1)
        transformer_output = self.transformer_encoder(x)
        transformer_embedding = torch.mean(transformer_output, dim=0)
        out = self.fc1_linear(transformer_embedding)
        return out
