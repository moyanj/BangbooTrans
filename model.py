# pylint:disable=W0612
import torch.nn as nn
import torch
import random
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)

    def forward(self, query, key, value, mask=None):
        attn_output, attn_weights = self.attn(query, key, value, attn_mask=mask)
        return attn_output, attn_weights


class EncoderRNN(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        hidden_dim,
        output_dim,
        num_layers,
        dropout,
        num_heads,
    ):
        super(EncoderRNN, self).__init__()

        self.hidden = None

        self.embedding = nn.Embedding(input_dim, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout
        )
        self.attention = SelfAttention(hidden_dim, num_heads)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):

        embedded = self.embedding(inputs)

        output, self.hidden = self.lstm(embedded)
        attn_output, attn_weights = self.attention(output, output, output)
        attn_output = self.fc(attn_output)
        return attn_output


class DecoderRNN(nn.Module):
    def __init__(
        self, input_dim, embed_dim, hidden_dim, output_dim, num_layers, dropout, n_heads
    ):
        super(DecoderRNN, self).__init__()

        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.attn = SelfAttention(hidden_dim, n_heads)
        self.out = nn.Linear(hidden_dim, output_dim)

        self.key_cache = None
        self.value_cache = None
        self.hidden = None

    def forward(self, target, encoder_outputs, step=0):
        target = target.unsqueeze(0)
        output = self.embedding(target)
        output, self.hidden = self.lstm(output, self.hidden)

        if step == 0:
            self.key_cache = encoder_outputs
            self.value_cache = encoder_outputs
        else:
            self.key_cache = torch.cat((self.key_cache, encoder_outputs), dim=0)
            self.value_cache = torch.cat((self.value_cache, encoder_outputs), dim=0)

        attn_output, attn_weights = self.attn(output, output, output)
        attn_output = attn_output.squeeze(0)
        prediction = self.out(attn_output)
        prediction = F.softmax(prediction, dim=1) # 将输出转换为概率分布 
        return prediction


class Bangboo(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.0):

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.out.out_features

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs = self.encoder(src)

        self.decoder.hidden = self.encoder.hidden

        inputs = trg[0, :].to(self.device)

        for t in range(1, trg_len):
            output = self.decoder(inputs, encoder_outputs, step=t - 1)
            outputs[t] = output
            top1 = output.argmax(1)
            inputs = trg[t] if random.random() < teacher_forcing_ratio else top1

        return outputs
