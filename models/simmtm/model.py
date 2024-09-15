import math

from torch import nn
import torch
from torch.nn import ModuleList

import torch.nn.functional as F

from models.simmtm.gtn.encoder import Encoder
from simmtm.loss import ContrastiveWeight, AggregationRebuild, AutomaticWeightedLoss


class TFC(nn.Module):
    def __init__(self, configs, args):

        super(TFC, self).__init__()
        self.training_mode = 'pre_train'

        # Projecting input into deep representations
        self.encoder_list_1 = ModuleList([Encoder(d_model=configs.d_model, d_hidden=configs.d_hidden, q=configs.q,
                                                  v=configs.v, h=configs.h, mask=configs.mask, dropout=configs.dropout,
                                                  device=configs.device) for _ in range(configs.N)])

        self.encoder_list_2 = ModuleList([Encoder(d_model=configs.d_model, d_hidden=configs.d_hidden, q=configs.q,
                                                  v=configs.v, h=configs.h, dropout=configs.dropout,
                                                  device=configs.device) for _ in range(configs.N)])

        self.embedding_channel = torch.nn.Linear(configs.d_channel, configs.d_model)
        self.embedding_input = torch.nn.Linear(configs.d_input, configs.d_model)

        self.gate = torch.nn.Linear(configs.d_model * configs.d_input + configs.d_model * configs.d_channel,
                                    configs.d_output)

        self.pe = configs.pe
        self._d_input = configs.d_input
        self._d_model = configs.d_model

        # MLP Layer - To generate Projector(.); to Obtain series-wise representations
        self.dense = nn.Sequential(
            nn.Linear(192512, 256),  # 240128 = encoder1 out features + encoder2 out features
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        if self.training_mode == 'pre_train':
            self.awl = AutomaticWeightedLoss(2)
            self.contrastive = ContrastiveWeight(args)
            self.aggregation = AggregationRebuild(args)
            # self.head = nn.Linear(240128, 336)  # Reconstruction, we have 336 time steps
            self.head = nn.Linear(192512, configs.d_input * configs.d_channel)  # Replaced to handle multi-variate
            self.mse = torch.nn.MSELoss()

    def forward(self, stage, x_in_t, pre_train=False):

        if pre_train:
            encoding_1 = self.embedding_channel(x_in_t)  # (128, 336, 512)
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
                encoding_1, score_input = encoder(encoding_1, stage)

            encoding_2 = self.embedding_input(x_in_t.transpose(-1, -2))
            channel_to_gather = encoding_2

            for encoder in self.encoder_list_2:
                encoding_2, score_channel = encoder(encoding_2, stage)

            encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
            encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)

            encoding_concat = self.gate(torch.cat([encoding_1, encoding_2], dim=-1))

            gate = F.softmax(encoding_concat, dim=-1)
            encoding = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1)

            # Projections
            projections = self.dense(encoding)

            loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(projections)
            rebuild_weight_matrix, agg_x = self.aggregation(similarity_matrix, encoding)

            pred_x = self.head(agg_x.reshape(agg_x.size(0), -1))

            loss_rb = self.mse(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())
            loss = self.awl(loss_cl, loss_rb)

            return loss, loss_cl, loss_rb

        else:
            encoding_1 = self.embedding_channel(x_in_t)  # (128, 336, 512)
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
                encoding_1, score_input = encoder(encoding_1, stage)

            encoding_2 = self.embedding_input(x_in_t.transpose(-1, -2))
            channel_to_gather = encoding_2

            for encoder in self.encoder_list_2:
                encoding_2, score_channel = encoder(encoding_2, stage)

            encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
            encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)

            encoding_concat = self.gate(torch.cat([encoding_1, encoding_2], dim=-1))

            gate = F.softmax(encoding_concat, dim=-1)
            encoding = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1)

            # Projections
            projections = self.dense(encoding)

            return encoding, projections


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
