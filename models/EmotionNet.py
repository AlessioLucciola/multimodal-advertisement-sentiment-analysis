import math
import torch
from torch import nn
from config import DROPOUT_P
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder


class EmotionNetTransformer(nn.Module):
    def __init__(self, num_classes, d_model=64, nhead=4, num_layers=6, dropout=DROPOUT_P):
        super(EmotionNetTransformer, self).__init__()
        self.num_classes = num_classes
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_model, dropout)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers)
        decoder_layers = TransformerDecoderLayer(
            d_model, nhead, d_model, dropout)
        self.transformer_decoder = TransformerDecoder(
            decoder_layers, num_layers)
        self.encoder = nn.Linear(2000, d_model)
        self.d_model = d_model
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        if self.src_mask is None or self.src_mask.size(0) != x.size(0):
            device = x.device
            mask = self._generate_square_subsequent_mask(x.size(0)).to(device)
            self.src_mask = mask
        x = self.encoder(x)
        x = x.unsqueeze(2)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, self.src_mask)
        x = self.transformer_decoder(
            tgt=x, memory=x, tgt_mask=self.src_mask, memory_mask=self.src_mask)
        # x = self.transformer_decoder(tgt=x, memory=self.src_mask)
        x = nn.Flatten()(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EmotionNet(nn.Module):
    def __init__(self, num_classes, dropout=DROPOUT_P):
        super(EmotionNet, self).__init__()
        self.num_classes = num_classes

        self.main_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),  # Add BatchNorm after Conv layer
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),  # Add BatchNorm after Conv layer
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Flatten(),
        )

        self.feature_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=1),
            nn.BatchNorm1d(32),  # Add BatchNorm after Conv layer
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),  # Add BatchNorm after Conv layer
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Flatten(),
        )

        self.transformer = EmotionNetTransformer(num_classes, dropout=dropout)

        # self.final_layer = nn.Linear(36160, num_classes * 2)
        self.final_layer = nn.Linear(35968, num_classes * 2)

    def forward(self, x, features=None):
        x_tr = x.clone()
        x = x.view(x.size(0), 1, -1)
        x_tr = self.transformer(x_tr)
        x = self.main_branch(x)
        # y = self.feature_branch(features)
        # out = torch.cat((x, x_tr, y), dim=1)
        out = torch.cat((x, x_tr), dim=1)
        out = self.final_layer(out)
        return out.view(out.size(0), 2, self.num_classes)
