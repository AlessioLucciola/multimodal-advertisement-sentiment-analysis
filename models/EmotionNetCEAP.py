from config import DROPOUT_P, LENGTH
import torch.nn as nn
import torch


class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(Encoder, self).__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

  def forward(self, x):
    # Pass data through LSTM layer
    x = x.unsqueeze(-1)
    lstm_out, (hidden, cell) = self.lstm(x)
    return hidden, cell  # Return the last hidden state and cell state

class Decoder(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(Decoder, self).__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    self.fc = nn.Linear(hidden_size, 1)

  def forward(self, x, hidden, cell):
    # Pass through decoder LSTM with previous hidden and cell state
    lstm_out, (hidden_new, cell_new) = self.lstm(x, (hidden, cell))
    # Pass through final dense layer for prediction
    # print(f"Decoder lstm_out shape: {lstm_out.shape}")
    prediction = self.fc(lstm_out[:, -1, :])
    # print(f"Decoder prediction shape: {prediction.shape}")
    return prediction, hidden_new, cell_new

class EmotionNet(nn.Module):
  def __init__(self, dropout=DROPOUT_P):
    super(EmotionNet, self).__init__()
    self.encoder = Encoder(input_size=1, hidden_size=32)
    self.decoder = Decoder(input_size=1, hidden_size=32)

  def forward(self, src, target):
      # Pass through encoder
    # Src shape must be (batch,seq,feature)
    # print(f"src shape: {src.shape}")
    hidden, cell = self.encoder(src)
    # print(f"Encoder hidden shape: {hidden.shape}, cell shape: {cell.shape}")
    # Pass through decoder with encoder hidden and cell state
    outputs = []
    for i in range(LENGTH):  # Assuming you know target sequence length
        curr_target = target[:, i].view(-1, 1, 1)
        # print(f"Current target shape: {curr_target.shape}")
        output, hidden, cell = self.decoder(curr_target, hidden, cell)
        outputs.append(output)
    return torch.stack(outputs, 1)  # Stack predictions for each time step

