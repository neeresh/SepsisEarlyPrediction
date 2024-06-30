import pandas as pd
import datetime

import os

import logging

import math

import torch

import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn import Module, ModuleList

from utils.path_utils import project_root

device = 'cuda'


def _setup_destination(current_time):
    log_path = os.path.join(project_root(), 'data', 'logs', current_time)
    os.mkdir(log_path)
    logging.basicConfig(filename=os.path.join(log_path, current_time + '.log'), level=logging.DEBUG)

    return log_path


def _log(message: str = '{}', value: any = None):
    print(message.format(value))
    logging.info(message.format(value))


def initialize_experiment(data_file):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    destination_path = _setup_destination(current_time)

    _log(message="Datafile used: {}".format(data_file))

    # [[patient1], [patient2], [patient3], ..., [patientN]]
    training_examples = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', data_file))
    _log(message="Total number of patients: {}", value=len(training_examples))

    with open(os.path.join(project_root(), 'data', 'processed', 'lengths.txt')) as f:
        lengths_list = [int(length) for length in f.read().splitlines()]
    _log(message="Min recordings: {} & Max recordings: {}".format(min(lengths_list), max(lengths_list)))

    with open(os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt')) as f:
        is_sepsis = [int(is_sep) for is_sep in f.read().splitlines()]
    _log(message="Distribution of the SepsisLabel: \n{}".format(pd.Series(is_sepsis).value_counts()))

    writer = SummaryWriter(log_dir=os.path.join(project_root(), 'data', 'logs', current_time), comment='')

    return training_examples, lengths_list, is_sepsis, writer, destination_path


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

        self.mask = True  # Masking padded rows by default

    def add_paddings(self, patient_data):
        max_time_step = 336
        current_length = patient_data.shape[1]  # Adjusted to take the correct dimension

        if current_length < max_time_step:
            pad_amount = max_time_step - current_length
            padding = torch.zeros((patient_data.shape[0], pad_amount, patient_data.shape[2]), dtype=patient_data.dtype,
                                  device=patient_data.device)
            patient_data = torch.cat((patient_data, padding), dim=1)

        return patient_data

    def forward(self, x_input, lengths, stage):

        # counter = 0
        # for t in range(1, time_steps+1):
        for t in range(0, lengths):
            x = x_input[:, :t, :]
            x = self.add_paddings(x)

            # Masking
            if self.mask:
                padding_mask = (x.sum(dim=-1) != 0)  # (batch_size, time_steps)
            else:
                padding_mask = None

            # step-wise
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
                encoding_1, score_input = encoder(encoding_1, stage, padding_mask=padding_mask)

            # channel-wise
            encoding_2 = self.embedding_input(x.transpose(-1, -2))
            channel_to_gather = encoding_2

            for encoder in self.encoder_list_2:
                encoding_2, score_channel = encoder(encoding_2, stage, padding_mask=None)

            encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
            encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)

            # gate
            gate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2], dim=-1)), dim=-1)
            encoding = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1)

            output = self.output_linear(encoding)  # Last output matters cuz, it sees entire trend
            # counter = counter + 1

        # print(counter)

        return output, encoding, score_input, score_channel, input_to_gather, channel_to_gather, gate
