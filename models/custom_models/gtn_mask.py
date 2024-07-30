import math

import torch
from torch import nn

import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, q: int, v: int, h: int, device: str, mask: bool = False, dropout: float = 0.1):
        super().__init__()
        super(MultiHeadAttention, self).__init__()

        self.w_q = nn.Linear(d_model, q * h)
        self.w_k = nn.Linear(d_model, q * h)
        self.w_v = nn.Linear(d_model, v * h)

        self.w_o = nn.Linear(v * h, d_model)
        self.device = device
        self._h = h
        self._q = q

        self.mask = mask
        self.dropout = nn.Dropout(p=dropout)
        self.score = None

    def forward(self, x, stage, padding_mask):  # Added mask here
        batch_size = x.size(0)
        seq_len = x.size(1)

        Q = torch.cat(self.w_q(x).chunk(self._h, dim=-1), dim=0)
        K = torch.cat(self.w_k(x).chunk(self._h, dim=-1), dim=0)
        V = torch.cat(self.w_v(x).chunk(self._h, dim=-1), dim=0)

        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        self.score = score

        # Masking future values to calculate attention scores
        if self.mask and stage == 'train':
            mask = torch.ones_like(score[0])
            mask = torch.tril(mask, diagonal=0)
            score = torch.where(mask > 0, score, torch.Tensor([-2 ** 32 + 1]).expand_as(score[0]).to(self.device))

        # Masking padded rows
        if padding_mask is not None:
            mask_expanded = padding_mask.unsqueeze(1).repeat(self._h, 1, 1).to(self.device)
            mask_expanded = mask_expanded.to(score.dtype)
            mask_expanded = mask_expanded.masked_fill(mask_expanded == 0, float('-inf'))

            score = score + mask_expanded

        score = F.softmax(score, dim=-1)

        attention = torch.matmul(score, V)
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)
        self_attention = self.w_o(attention_heads)

        return self_attention, self.score


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = 512):
        super(FeedForward, self).__init__()

        self.linear_1 = torch.nn.Linear(d_model, d_hidden)
        self.linear_2 = torch.nn.Linear(d_hidden, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, q: int, v: int, h: int, device: str, mask: bool = False,
                 dropout: float = 0.1):
        super(Encoder, self).__init__()
        self.mha = MultiHeadAttention(d_model=d_model, q=q, v=v, h=h, mask=mask, device=device, dropout=dropout)
        self.feedforward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x, stage, mask):  # Added mask here
        residual = x
        x, score = self.mha(x, stage, mask)  # Added mask here
        x = self.dropout(x)
        x = self.layer_norm1(x + residual)

        residual = x
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.layer_norm2(x + residual)

        return x, score


class MaskedGatedTransformerNetwork(nn.Module):
    def __init__(self, d_model: int, d_input: int, d_channel: int, d_output: int, d_hidden: int, q: int, v: int,
                 h: int, N: int, device: str, dropout: float = 0.1, pe: bool = False, mask: bool = False):
        super(MaskedGatedTransformerNetwork, self).__init__()

        self.encoder_list_1 = nn.ModuleList([Encoder(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h, mask=mask,
                                                     dropout=dropout, device=device) for _ in range(N)])
        self.encoder_list_2 = nn.ModuleList([Encoder(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h, mask=mask,
                                                     dropout=dropout, device=device) for _ in range(N)])

        self.embedding_channel = nn.Linear(d_channel, d_model)
        self.embedding_input = nn.Linear(d_input, d_model)

        self.gate = nn.Linear(d_model * d_input + d_model * d_channel, 2)
        self.output_linear = nn.Linear(d_model * d_input + d_model * d_channel, d_output)

        self.pe = pe
        self._d_input = d_input
        self._d_model = d_model

    def forward(self, x, stage, mask):  # Added mask here

        encoding_1 = self.embedding_channel(x)
        input_to_gather = encoding_1

        if self.pe:
            pe = torch.ones_like(encoding_1[0])
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            encoding_1 = encoding_1 + pe

        for encoder in self.encoder_list_1:
            encoding_1, score_input = encoder(encoding_1, stage, mask)  # Added mask here

        encoding_2 = self.embedding_input(x.transpose(-1, -2))
        channel_to_gather = encoding_2

        for encoder in self.encoder_list_2:
            encoding_2, score_channel = encoder(encoding_2, stage, None)

        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)

        gate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2], dim=-1)), dim=-1)
        encoding = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1)

        output = self.output_linear(encoding)

        return output, encoding, score_input, score_channel, input_to_gather, channel_to_gather, gate
