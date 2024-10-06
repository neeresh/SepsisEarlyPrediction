import math
from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import ModuleList

from models.tfc.gtn.encoder import Encoder


class TFC(nn.Module):
    def __init__(self, configs, args):

        super(TFC, self).__init__()
        self.training_mode = 'pre_train'

        """ TIME """
        # Projecting input into deep representations
        self.encoder_list_1_t = ModuleList([Encoder(d_model=configs.d_model, d_hidden=configs.d_hidden, q=configs.q,
                                                  v=configs.v, h=configs.h, mask=configs.mask, dropout=configs.dropout,
                                                  device=configs.device) for _ in range(configs.N)])

        self.encoder_list_2_t = ModuleList([Encoder(d_model=configs.d_model, d_hidden=configs.d_hidden, q=configs.q,
                                                  v=configs.v, h=configs.h, dropout=configs.dropout,
                                                  device=configs.device) for _ in range(configs.N)])

        self.embedding_channel_t = torch.nn.Linear(configs.d_channel, configs.d_model)
        self.embedding_input_t = torch.nn.Linear(configs.d_input, configs.d_model)

        self.gate_t = torch.nn.Linear(configs.d_model * configs.d_input + configs.d_model * configs.d_channel,
                                    configs.d_output)

        self.pe_t = configs.pe
        self._d_input_t = configs.d_input
        self._d_model_t = configs.d_model

        # Adding MLP
        gate_output_dim_t = 192512
        self.mlp_fc1_t = torch.nn.Linear(gate_output_dim_t, int(gate_output_dim_t / 64))
        self.mlp_relu_t = torch.nn.ReLU()
        self.mlp_fc2_t = torch.nn.Linear(int(gate_output_dim_t / 64), int(gate_output_dim_t / 64))

        # MLP Layer - To generate Projector(.); to Obtain series-wise representations
        self.dense_t = nn.Sequential(
            nn.Linear(int(gate_output_dim_t / 64), 256),  # 240128 = encoder1 out features + encoder2 out features
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        """FREQUENCY"""
        # Projecting input into deep representations
        self.encoder_list_1_f = ModuleList([Encoder(d_model=configs.d_model, d_hidden=configs.d_hidden, q=configs.q,
                                                  v=configs.v, h=configs.h, mask=configs.mask, dropout=configs.dropout,
                                                  device=configs.device) for _ in range(configs.N)])

        self.encoder_list_2_f = ModuleList([Encoder(d_model=configs.d_model, d_hidden=configs.d_hidden, q=configs.q,
                                                  v=configs.v, h=configs.h, dropout=configs.dropout,
                                                  device=configs.device) for _ in range(configs.N)])

        self.embedding_channel_f = torch.nn.Linear(configs.d_channel, configs.d_model)
        self.embedding_input_f = torch.nn.Linear(configs.d_input, configs.d_model)

        self.gate_f = torch.nn.Linear(configs.d_model * configs.d_input + configs.d_model * configs.d_channel,
                                    configs.d_output)

        self.pe_f = configs.pe
        self._d_input_f = configs.d_input
        self._d_model_f = configs.d_model

        # Adding MLP
        gate_output_dim_f = 192512
        self.mlp_fc1_f = torch.nn.Linear(gate_output_dim_f, int(gate_output_dim_f / 64))
        self.mlp_relu_f = torch.nn.ReLU()
        self.mlp_fc2_f = torch.nn.Linear(int(gate_output_dim_f / 64), int(gate_output_dim_f / 64))

        # MLP Layer - To generate Projector(.); to Obtain series-wise representations
        self.dense_f = nn.Sequential(
            nn.Linear(int(gate_output_dim_f / 64), 256),  # 240128 = encoder1 out features + encoder2 out features
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def _process_x_in_t(self, x_in_t, stage):

        encoding_1_t = self.embedding_channel_t(x_in_t)  # (128, 336, 512)

        if self.pe:
            pe_t = torch.ones_like(encoding_1_t[0])
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe_t[:, 0::2] = torch.sin(temp)
            pe_t[:, 1::2] = torch.cos(temp)

            encoding_1_t = encoding_1_t + pe_t

        for encoder_t in self.encoder_list_1_t:
            encoding_1_t, score_input_t = encoder_t(encoding_1_t, stage)

        encoding_2_t = self.embedding_input_t(x_in_t.transpose(-1, -2))

        for encoder_t in self.encoder_list_2_t:
            encoding_2_t, score_channel_t = encoder_t(encoding_2_t, stage)

        encoding_1_t = encoding_1_t.reshape(encoding_1_t.shape[0], -1)
        encoding_2_t = encoding_2_t.reshape(encoding_2_t.shape[0], -1)

        encoding_concat_t = self.gate_t(torch.cat([encoding_1_t, encoding_2_t], dim=-1))

        gate_t = F.softmax(encoding_concat_t, dim=-1)
        encoding_t = torch.cat([encoding_1_t * gate_t[:, 0:1], encoding_2_t * gate_t[:, 1:2]], dim=-1)

        # Adding MLP
        encoding_t = self.mlp_fc1_t(encoding_t)
        encoding_t = self.mlp_relu_t(encoding_t)
        encoding_t = self.mlp_fc2_t(encoding_t)

        # Projections
        projections_t = self.dense_t(encoding_t)

        return encoding_t, projections_t

    def _process_x_in_f(self, x_in_f, stage):

        encoding_1_f = self.embedding_channel_f(x_in_f)  # (128, 336, 512)

        if self.pe:
            pe_f = torch.ones_like(encoding_1_f[0])
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe_f[:, 0::2] = torch.sin(temp)
            pe_f[:, 1::2] = torch.cos(temp)

            encoding_1_f = encoding_1_f + pe_f

        for encoder_f in self.encoder_list_1_f:
            encoding_1_f, score_input_f = encoder_f(encoding_1_f, stage)

        encoding_2_f = self.embedding_input_f(x_in_f.transpose(-1, -2))

        for encoder_f in self.encoder_list_2_f:
            encoding_2_f, score_channel_f = encoder_f(encoding_2_f, stage)

        encoding_1_f = encoding_1_f.reshape(encoding_1_f.shape[0], -1)
        encoding_2_f = encoding_2_f.reshape(encoding_2_f.shape[0], -1)

        encoding_concat_f = self.gate_f(torch.cat([encoding_1_f, encoding_2_f], dim=-1))

        gate_f = F.softmax(encoding_concat_f, dim=-1)
        encoding_f = torch.cat([encoding_1_f * gate_f[:, 0:1], encoding_2_f * gate_f[:, 1:2]], dim=-1)

        # Adding MLP
        encoding_f = self.mlp_fc1_f(encoding_f)
        encoding_f = self.mlp_relu_f(encoding_f)
        encoding_f = self.mlp_fc2_f(encoding_f)

        # Projections
        projections_f = self.dense_f(encoding_f)

        return encoding_f, projections_f

    def forward(self, x_in_t, x_in_f, stage):
        h_time, z_time = self._process_x_in_t(stage, x_in_t)
        h_freq, z_freq = self._process_x_in_t(stage, x_in_f)

        return h_time, z_time, h_freq, z_freq


class target_classifier(nn.Module):  # Classification head
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(192512, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        """2-layer MLP"""
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred
