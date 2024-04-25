from config import DROPOUT_P, LENGTH, LSTM_DEC_HIDDEN,LSTM_ENC_HIDDEN
import torch.nn as nn
import torch


class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(Encoder, self).__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    self.dropout = nn.Dropout(p=DROPOUT_P)

  def forward(self, x):
    if x.ndim < 3:
        x = x.unsqueeze(-1)
    lstm_out, (hidden, cell) = self.lstm(x)
    hidden = self.dropout(hidden)
    return hidden, cell  

class Decoder(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(Decoder, self).__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    num_classes = 3
    self.fc = nn.Linear(hidden_size, num_classes)
    self.dropout = nn.Dropout(p=DROPOUT_P)

  def forward(self, x, hidden, cell):
    lstm_out, (hidden_new, cell_new) = self.lstm(x, (hidden, cell))
    lstm_out = self.dropout(lstm_out)
    prediction = self.fc(lstm_out[:, -1, :])
    return prediction, hidden_new, cell_new

class EmotionNet(nn.Module):
  def __init__(self, dropout=DROPOUT_P):
    super(EmotionNet, self).__init__()
    self.encoder = Encoder(input_size=1, hidden_size=LSTM_ENC_HIDDEN)
    self.decoder = Decoder(input_size=1, hidden_size=LSTM_DEC_HIDDEN)

  def forward(self, src, target):
    hidden, cell = self.encoder(src)
    # Pass through decoder with encoder hidden and cell state
    outputs = []
    for i in range(LENGTH):  # Assuming you know target sequence length
        curr_target = target[:, i].view(-1, 1, 1)
        # print(f"Current target shape: {curr_target.shape}")
        output, hidden, cell = self.decoder(curr_target, hidden, cell)
        outputs.append(output)
    return torch.stack(outputs, 1)  # Stack predictions for each time step

