import math

import torch

import torch.nn.functional as F
from torch.nn import Module, ModuleList


class MultiHeadAttention(Module):
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool = False,
                 dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.W_q = torch.nn.Linear(d_model, q * h)
        self.W_k = torch.nn.Linear(d_model, q * h)
        self.W_v = torch.nn.Linear(d_model, v * h)

        self.W_o = torch.nn.Linear(v * h, d_model)

        self.device = device
        self._h = h
        self._q = q

        self.mask = mask
        self.dropout = torch.nn.Dropout(p=dropout)
        self.score = None

    def forward(self, x, stage, padding_mask):
        Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0)
        K = torch.cat(self.W_k(x).chunk(self._h, dim=-1), dim=0)
        V = torch.cat(self.W_v(x).chunk(self._h, dim=-1), dim=0)

        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        self.score = score

        if self.mask and stage == 'train':
            mask = torch.ones_like(score[0])
            mask = torch.tril(mask, diagonal=0)
            score = torch.where(mask > 0, score, torch.Tensor([-2 ** 32 + 1]).expand_as(score[0]).to(self.device))

        # Masking Paddings
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)

            padding_mask = padding_mask.expand(-1, self._h, score.size(-1), score.size(-1))
            padding_mask = padding_mask.contiguous().view(-1, score.size(-1), score.size(-1))

            score = score.masked_fill(padding_mask == 0, -2 ** 32 + 1)

        score = F.softmax(score, dim=-1)

        attention = torch.matmul(score, V)

        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        self_attention = self.W_o(attention_heads)

        return self_attention, self.score


class FeedForward(Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int = 512):
        super(FeedForward, self).__init__()

        self.linear_1 = torch.nn.Linear(d_model, d_hidden)
        self.linear_2 = torch.nn.Linear(d_hidden, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)

        return x


class Encoder(Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool = False,
                 dropout: float = 0.1):
        super(Encoder, self).__init__()

        self.MHA = MultiHeadAttention(d_model=d_model, q=q, v=v, h=h, mask=mask, device=device, dropout=dropout)
        self.feedforward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layerNormal_1 = torch.nn.LayerNorm(d_model)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model)

    def forward(self, x, stage, padding_mask=None):
        residual = x
        x, score = self.MHA(x, stage, padding_mask)
        x = self.dropout(x)
        x = self.layerNormal_1(x + residual)

        residual = x
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.layerNormal_2(x + residual)

        return x, score


class ModifiedGatedTransformerNetwork(Module):
    def __init__(self,
                 d_model: int,
                 d_input: int,
                 d_channel: int,
                 d_output: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 device: str,
                 dropout: float = 0.1,
                 pe: bool = False,
                 mask: bool = False):
        super(ModifiedGatedTransformerNetwork, self).__init__()

        self.encoder_list_1 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  mask=mask,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.encoder_list_2 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.embedding_channel = torch.nn.Linear(d_channel, d_model)
        self.embedding_input = torch.nn.Linear(d_input, d_model)

        self.gate = torch.nn.Linear(d_model * d_input + d_model * d_channel, 2)
        self.output_linear = torch.nn.Linear(d_model * d_input + d_model * d_channel, d_output)

        self.pe = pe
        self._d_input = d_input
        self._d_model = d_model
        self.d_output = d_output

        self.mask = True  # Masking padded rows by default

    def add_paddings(self, x, max_length):
        batch_size, current_length, channels = x.shape
        if current_length < max_length:
            padding = torch.zeros((batch_size, max_length - current_length, channels), dtype=x.dtype, device=x.device)
            x = torch.cat((x, padding), dim=1)
        return x

    def forward(self, x_input, lengths, stage):
        batch_size, time_steps, channels = x_input.size()
        max_time_step = 336  # Define the maximum time steps based on your data

        # Initialize outputs
        final_outputs = torch.zeros(batch_size, self.d_output, device=x_input.device)

        # Iterate over each sample in the batch
        for sample_id in range(batch_size):
            x_input_single, lengths_single = x_input[sample_id].unsqueeze(0), lengths[sample_id]

            # Iterate over each time step for the current sample
            for t in range(lengths_single):
                x = x_input_single[:, :t + 1, :]  # Select data from the first time step to the current time step

                # Add padding if necessary
                x = self.add_paddings(x, max_time_step)

                # Masking
                if self.mask:
                    padding_mask = (x.sum(dim=-1) != 0)  # (batch_size, time_steps)
                else:
                    padding_mask = None

                # Step-wise encoding
                encoding_1 = self.embedding_channel(x)
                if self.pe:
                    position = torch.arange(0, max_time_step, device=x_input.device).unsqueeze(1)
                    div_term = torch.exp(torch.arange(0, self._d_model, 2, device=x_input.device) *
                                         -(math.log(10000.0) / self._d_model))
                    pe = torch.zeros(max_time_step, self._d_model, device=x_input.device)
                    pe[:, 0::2] = torch.sin(position * div_term)
                    pe[:, 1::2] = torch.cos(position * div_term)
                    encoding_1 += pe

                for encoder in self.encoder_list_1:
                    encoding_1, score_input = encoder(encoding_1, stage, padding_mask=padding_mask)

                # Channel-wise encoding
                encoding_2 = self.embedding_input(x.transpose(-1, -2))
                for encoder in self.encoder_list_2:
                    encoding_2, score_channel = encoder(encoding_2, stage)

                encoding_1 = encoding_1.view(1, -1)
                encoding_2 = encoding_2.view(1, -1)

                # Gate
                gate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2], dim=-1)), dim=-1)
                encoding = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1)

                output = self.output_linear(encoding)  # Output after processing the entire sequence up to t

            # Storing output after all the time steps are executed
            final_outputs[sample_id] = output

        return final_outputs, encoding, score_input, score_channel, encoding_1, encoding_2, gate
