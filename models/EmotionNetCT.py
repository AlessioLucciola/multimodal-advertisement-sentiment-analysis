from config import DROPOUT_P, T_HEAD, T_ENC_LAYERS, T_DIM_FFW, T_KERN, T_STRIDE, T_MAXPOOL, LENGTH, WAVELET_STEP, BATCH_SIZE
import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# class Block(nn.Module):
#     def __init__(self, d_model, dropout):
#         super(Block, self).__init__()
#         self.attention = nn.MultiheadAttention(d_model, T_HEAD)
#         self.ffn = nn.Sequential(
#                     nn.Linear(d_model, 2 * d_model),
#                     nn.LeakyReLU(),
#                     nn.Linear(2 * d_model, d_model))
#         self.drop1 = nn.Dropout(dropout)
#         self.drop2 = nn.Dropout(dropout)
#         self.ln1 = nn.LayerNorm(d_model)
#         self.ln2 = nn.LayerNorm(d_model)

#     def forward(self, hidden_state):
#         attn, _ = self.attention(hidden_state, hidden_state, hidden_state, need_weights=False)
#         attn = self.drop1(attn)
#         out = self.ln1(hidden_state + attn)
#         observed = self.ffn(out)
#         observed = self.drop2(observed)
#         return self.ln2(out + observed)

class EmotionNet(nn.Module):
    def __init__(self, num_classes, dropout=DROPOUT_P):
        super().__init__()
        self.transformer_maxpool = nn.MaxPool2d(
                kernel_size=[1, T_KERN], stride=[1, T_STRIDE])


        d_model = 256
        print(f"d_model is: {d_model}")
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        t_dropout = 0.3
        transformer_enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=T_HEAD,
                dim_feedforward=T_DIM_FFW,
                dropout=t_dropout,
                activation='relu')

        self.transformer_encoder = nn.TransformerEncoder(
                transformer_enc_layer, num_layers=T_ENC_LAYERS)

        # self.transformer_encoder = Block(d_model=d_model, 
        #                                  dropout=dropout)
        
        self.fc1_linear = nn.Sequential(nn.Linear(d_model, d_model),
			nn.LeakyReLU(),
			nn.Dropout(dropout),
			nn.Linear(d_model, d_model),
			nn.LeakyReLU(),
			nn.Dropout(dropout),
			nn.Linear(d_model, num_classes))

        # self.fc1_linear = nn.Linear(d_model, num_classes)

    def forward(self, _, x):
        if T_MAXPOOL != 0:
            x = x.unsqueeze(1)
        for _ in range(T_MAXPOOL):
            x = self.transformer_maxpool(x)
            x = torch.squeeze(x, 1)

        # print(f"x shape is: {x.shape}")
        x = x.permute(2, 0, 1)
        x = self.pos_encoder(x)
        embedding = self.transformer_encoder(x)
        # print(f"transformer output shape is {embedding.shape}")
        embedding = torch.mean(embedding, dim=0)

        out = self.fc1_linear(embedding)
        return out
