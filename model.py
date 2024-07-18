import torch.nn as nn
import torch
import config

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden2_size, num_layers):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.hidden2_size = hidden2_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden2_size)
        self.lstm = nn.LSTM(hidden2_size, hidden_size, num_layers=num_layers)

    def forward(self, inputs, hidden=None):
        if hidden is None:
            hidden = self.init_hidden()

        embedded = self.embedding(inputs).view(1, 1, -1).to(config.device)
        output, hidden = self.lstm(embedded, hidden)

        return output, hidden

    def init_hidden(self):
        return (
            torch.zeros(self.num_layers, 1, self.hidden_size, device=config.device),
            torch.zeros(self.num_layers, 1, self.hidden_size, device=config.device),
        )


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, hidden2_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, hidden2_size)
        self.lstm = nn.LSTM(hidden2_size, hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden()

        output = self.embedding(input).view(1, 1, -1).to(config.device)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return (
            torch.zeros(self.num_layers, 1, self.hidden_size, device=config.device),
            torch.zeros(self.num_layers, 1, self.hidden_size, device=config.device),
        )


criterion = nn.NLLLoss()
