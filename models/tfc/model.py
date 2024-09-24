import math
from torch import nn
import torch
import torch.nn.functional as F
from models.tfc.gtn.encoder import Encoder


class TFC(nn.Module):
    def __init__(self, configs):
        super(TFC, self).__init__()
        self.training_mode = 'pre_train'

        # Time and Frequency Encoders
        self.encoder_list_1_t = self._create_encoder_list(configs, domain='t')
        self.encoder_list_1_f = self._create_encoder_list(configs, domain='f')

        self.encoder_list_2_t = self._create_encoder_list(configs, domain='t')
        self.encoder_list_2_f = self._create_encoder_list(configs, domain='f')

        # Embeddings for Time and Frequency
        self.embedding_channel_t = self._create_embedding_layer(configs, domain='t')
        self.embedding_channel_f = self._create_embedding_layer(configs, domain='f')

        self.embedding_input_t = self._create_input_embedding_layer(configs, domain='t')
        self.embedding_input_f = self._create_input_embedding_layer(configs, domain='f')

        # Gates
        self.gate_t = self._create_gate_layer(configs, domain='t')
        self.gate_f = self._create_gate_layer(configs, domain='f')

        # Positional Encoding
        self.pe_t = configs.pe
        self.pe_f = configs.pe

        # Projector
        self.projector_t = nn.Sequential(
            nn.Linear(192512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.projector_f = nn.Sequential(
            nn.Linear(192512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self._d_input_t = configs.d_input
        self._d_input_f = configs.d_input

        self._d_model_t = configs.d_model
        self._d_model_f = configs.d_model

    # Helper function to create encoders dynamically for time or frequency
    def _create_encoder_list(self, configs, domain):
        return nn.ModuleList([Encoder(d_model=configs.d_model, d_hidden=configs.d_hidden, q=configs.q,
                                      v=configs.v, h=configs.h, mask=configs.mask, dropout=configs.dropout,
                                      device=configs.device) for _ in range(configs.N)])

    def _create_embedding_layer(self, configs, domain):
        return nn.Linear(configs.d_channel, configs.d_model)

    def _create_input_embedding_layer(self, configs, domain):
        return nn.Linear(configs.d_input, configs.d_model)

    def _create_gate_layer(self, configs, domain):
        return nn.Linear(configs.d_model * configs.d_input + configs.d_model * configs.d_channel, configs.d_output)

    def forward(self, x_in_t, x_in_f, stage):

        # Time Domain Processing
        h_time, z_time = self._process_domain(x_in_t, stage, domain='t')

        # Frequency Domain Processing
        h_freq, z_freq = self._process_domain(x_in_f, stage, domain='f')

        return h_time, z_time, h_freq, z_freq

    def _process_domain(self, x_in, stage, domain):

        # x_in = x_in.permute(0, 2, 1)

        encoding_1 = self._get_embedding_channel(x_in, domain)
        input_to_gather = encoding_1

        if getattr(self, f"pe_{domain}"):
            encoding_1 = self._apply_positional_encoding(encoding_1, domain)

        for encoder in getattr(self, f'encoder_list_1_{domain}'):
            encoding_1, score_input = encoder(encoding_1, stage)

        encoding_2 = self._get_embedding_input(x_in.transpose(-1, -2), domain)
        channel_to_gather = encoding_2

        for encoder in getattr(self, f'encoder_list_2_{domain}'):
            encoding_2, score_channel = encoder(encoding_2, stage)

        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)

        encoding_concat = getattr(self, f'gate_{domain}')(torch.cat([encoding_1, encoding_2], dim=-1))
        gate = F.softmax(encoding_concat, dim=-1)
        h = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1)
        z = getattr(self, f'projector_{domain}')(h)

        return h, z

    def _get_embedding_channel(self, x_in, domain):
        return getattr(self, f'embedding_channel_{domain}')(x_in)

    def _get_embedding_input(self, x_in, domain):
        return getattr(self, f'embedding_input_{domain}')(x_in)

    def _apply_positional_encoding(self, encoding_1, domain):
        pe = torch.ones_like(encoding_1[0])
        position = torch.arange(0, getattr(self, f'_d_input_{domain}')).unsqueeze(-1)
        temp = torch.Tensor(range(0, getattr(self, f'_d_model_{domain}'), 2))
        temp = temp * -(math.log(10000) / getattr(self, f'_d_model_{domain}'))
        temp = torch.exp(temp).unsqueeze(0)
        temp = torch.matmul(position.float(), temp)
        pe[:, 0::2] = torch.sin(temp)
        pe[:, 1::2] = torch.cos(temp)

        return encoding_1 + pe


"""Downstream classifier only used in finetuning"""


class target_classifier(nn.Module):
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(512, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)

        return pred
