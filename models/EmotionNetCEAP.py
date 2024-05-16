from config import WT
import torch.nn as nn
import torch
from utils.utils import select_device
from typing import Optional
import random

# Code bootstraped from "https://github.com/ethen8181/machine-learning/blob/master/deep_learning/seq2seq/1_torch_seq2seq_intro.ipynb"
# TODO: The above is not so true, but I couldn't find the original source. The followind source is pretty similar though.

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(input_dim, embedding_dim)

    def forward(self, src):
        # src = [src length, batch size]
        if not WT:  
            src = src.unsqueeze(-1)
        embedded = src
        # embedded = self.embedding(src)
        # embedded = [src length, batch size, embedding dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(3, embedding_dim)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hiden dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hidden dim]
        # context = [n layers, batch size, hidden dim]
        # input = [1, batch size] #OK
        #TODO: I don't know about this repeat
        # old_embedded = input.view(1, -1, 1).repeat(1, 1, self.embedding_dim)

        embedded = self.embedding(input.to(torch.long)).unsqueeze(0)
        # print(f"embedded shape is: {embedded.shape}, old_embedded shape is {old_embedded.shape}")
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq length, batch size, hidden dim * n directions]emotion
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # seq length and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim] #OK
        # hidden = [n layers, batch size, hidden dim] #OK
        # cell = [n layers, batch size, hidden dim] #OK
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim] #OK
        return prediction, hidden, cell

class EmotionNet(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = select_device()
        self.tf_ratio = 0.5
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"
    
    def train(self, mode: bool = True):
        self.curr_tf_ratio = self.tf_ratio if mode else 0
        print(f"EmotionNet teacher forcing ratio is set to {self.curr_tf_ratio}")
        return super().train(mode)
    
    def forward(self, src, trg, memory: Optional[tuple] = None, first_input: Optional[torch.Tensor] = None):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = src.shape[1]
        trg_length = src.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        if memory is None:
            hidden, cell = self.encoder(src)
        else:
            hidden, cell = memory
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        if first_input is not None:
            input = first_input
        else:
            # if len(trg.shape) == 2:
            # elif len(trg.shape) == 1:
            #     input = trg.view(1, -1)
            # else:
            #     raise ValueError(f"trg with shape: {trg.shape} is not supported")
            # print(f"input is: {input}")
            # if self.training:
            # input = trg[0,:]
            # else:
            input = torch.full((batch_size,), -1).to(self.device) #SOS token
            # print(f"input shape is: {input.shape}")
            # input = [batch size]
        for t in range(1, trg_length):
            # self.adjust_tf_ratio(t) 
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            # output = [batch size, output dim] #OK
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            # get the highest predicted token from our predictions
            if random.random() < self.curr_tf_ratio:
                input = trg[t]
            else:
                input = output.argmax(1).float()
            # input = [batch size]
        return outputs, (hidden, cell)



def add_noise(targets: torch.Tensor, noise_factor=0.1) -> torch.Tensor:
  """
  Adds random noise to a batched tensor of target labels (integers between 0 and 2).

  Args:
      targets: A batched tensor of target labels (long type) with size (batch_size).
      noise_factor: The proportion of the range to use for noise (default: 0.1).

  Returns:
      A batched tensor of noised target labels (long type) with size (batch_size).
  """
  # Check if targets is long type
  # if not targets.dtype == torch.long:
  #   raise ValueError("targets must be a long tensor (dtype=torch.long)")

  # Calculate the noise range based on noise_factor
  noise_range = (targets.max() - targets.min()) * noise_factor

  # Generate random noise tensor with same size as targets
  noise = torch.rand_like(targets) * noise_range

  # Clip noise values between 0 and noise_range
  noise = torch.clamp(noise, min=0, max=noise_range)

  # Add noise to targets while keeping integer type (round)
  noised_targets = torch.round(targets.float() + noise).long()

  # Clip noised_targets values between 0 and the maximum target value
  noised_targets = torch.clamp(noised_targets, min=0, max=targets.max())

  return noised_targets
