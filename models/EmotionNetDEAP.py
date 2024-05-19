
from config import DROPOUT_P, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, EMOTION_NUM_CLASSES
import torch.nn as nn
import torch
import torch.nn.functional as F

class EmotionNet(nn.Module):
    def __init__(self, num_classes, input_size, num_layers=LSTM_NUM_LAYERS, hidden_size=LSTM_HIDDEN_SIZE, dropout_p=DROPOUT_P):
        super().__init__() 
        input_size = 128
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            # dropout=dropout_p,
                            bidirectional=False)
        self.CNN_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(
                in_channels=8, 
                out_channels=16,
                kernel_size=3,
                stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=dropout_p), 
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=dropout_p),
        )

        self.CNN_block_1d = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
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
            nn.Flatten()
        )

        # self.fc = nn.Sequential(
        #         nn.Linear(256, 128),
        #         nn.ReLU(),
        #         nn.Dropout(0.5),
        #         nn.Linear(128, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, EMOTION_NUM_CLASSES))
        self.fc = nn.Linear(7936, EMOTION_NUM_CLASSES)

    def forward(self, x):
        x = x.unsqueeze(1)
        final= self.CNN_block_1d(x)
        # CNN_embedding = torch.flatten(CNN_embedding, start_dim=1) 

        # x_lstm = CNN_embedding.permute(0, 2, 1)

        # Should be (batch, seq, feature)
        # print(f"lstm x shape is: {x_lstm.shape}")

        # lstm_output, _ = self.lstm(x_lstm)
        # lstm_embedding = lstm_output[:, -1, :]
        # final = torch.flatten(lstm_embedding, start_dim=1) 

        # complete_embedding = torch.cat([CNN_embedding, lstm_embedding], dim=1)
        # complete_embedding = lstm_embedding
        out = self.fc(final)  
        # print(f"output logits are: {output_logits}")
        # print(f"preds are: {output_logits.argmax(1)}")
        # return F.log_softmax(output_logits, dim=-1)
        return out
