import math
from torch import nn
import torch
from torch.autograd import Function
from torch.nn import ModuleList

from models.adatime.gtn.encoder import Encoder
import torch.nn.functional as F


def get_backbone_class(backbone_name):
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


class GTN(nn.Module):

    def __init__(self, configs):
        super(GTN, self).__init__()

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

        # self.head = nn.Linear(192512, int((configs.d_input * configs.d_channel)/4))

    def forward(self, stage, x_in_t):

        encoding_1 = self.embedding_channel(x_in_t)

        if self.pe:
            pe = torch.ones_like(encoding_1[0])
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            encoding_1 = encoding_1 + pe

        for encoder in self.encoder_list_1:
            encoding_1, score_input = encoder(encoding_1, stage)

        encoding_2 = self.embedding_input(x_in_t.transpose(-1, -2))

        for encoder in self.encoder_list_2:
            encoding_2, score_channel = encoder(encoding_2, stage)

        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)

        encoding_concat = self.gate(torch.cat([encoding_1, encoding_2], dim=-1))

        gate = F.softmax(encoding_concat, dim=-1)
        encoding = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1)
        # encoding = self.head(encoding)

        return encoding


class classifier(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()
        self.logits = nn.Linear(configs.d_model * configs.d_input + configs.d_model * configs.d_channel,
                                configs.num_classes)
        self.configs = configs

    def forward(self, x):
        predictions = self.logits(x)

        return predictions


#### Codes required by DANN ##############
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            # nn.Linear(configs.features_len * configs.final_out_channels, configs.dann_disc_hid_dim),
            nn.Linear(configs.d_model * configs.d_input + configs.d_model * configs.d_channel,
                      configs.dann_disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.dann_disc_hid_dim, configs.dann_disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.dann_disc_hid_dim, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


#### Codes required by CDAN ##############
class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


class Discriminator_CDAN(nn.Module):
    """Discriminator model for CDAN ."""

    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator_CDAN, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(385024, configs.cdan_disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.cdan_disc_hid_dim, configs.cdan_disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.cdan_disc_hid_dim, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


class codats_classifier(nn.Module):
    def __init__(self, configs):
        super(codats_classifier, self).__init__()
        model_output_dim = configs.features_len
        self.hidden_dim = configs.codats_hidden_dim
        self.logits = nn.Sequential(
            nn.Linear(configs.d_model * configs.d_input + configs.d_model * configs.d_channel, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, configs.num_classes))

    def forward(self, x_in):
        predictions = self.logits(x_in)
        return predictions


#### Codes required by AdvSKM ##############
class Cosine_act(nn.Module):
    def __init__(self):
        super(Cosine_act, self).__init__()

    def forward(self, input):
        return torch.cos(input)


cos_act = Cosine_act()


class AdvSKM_Disc(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs):
        """Init discriminator."""
        super(AdvSKM_Disc, self).__init__()

        self.input_dim = configs.d_model * configs.d_input + configs.d_model * configs.d_channel
        self.hid_dim = configs.advskm_DSKN_disc_hid
        self.branch_1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            cos_act,
            nn.Linear(self.hid_dim, self.hid_dim // 2),
            nn.Linear(self.hid_dim // 2, self.hid_dim // 2),
            nn.BatchNorm1d(self.hid_dim // 2),
            cos_act
        )
        self.branch_2 = nn.Sequential(
            nn.Linear(configs.d_model * configs.d_input + configs.d_model * configs.d_channel,
                      configs.advskm_disc_hid_dim),
            nn.Linear(configs.advskm_disc_hid_dim, configs.advskm_disc_hid_dim),
            nn.BatchNorm1d(configs.advskm_disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.advskm_disc_hid_dim, configs.advskm_disc_hid_dim // 2),
            nn.Linear(configs.advskm_disc_hid_dim // 2, configs.advskm_disc_hid_dim // 2),
            nn.BatchNorm1d(configs.advskm_disc_hid_dim // 2),
            nn.ReLU())

    def forward(self, input):
        """Forward the discriminator."""
        out_cos = self.branch_1(input)
        out_rel = self.branch_2(input)
        total_out = torch.cat((out_cos, out_rel), dim=1)
        return total_out


class attn_network(nn.Module):
    def __init__(self, configs):
        super(attn_network, self).__init__()

        self.h_dim = configs.features_len * configs.final_out_channels
        self.self_attn_Q = nn.Sequential(nn.Linear(in_features=self.h_dim, out_features=self.h_dim),
                                         nn.ELU()
                                         )
        self.self_attn_K = nn.Sequential(nn.Linear(in_features=self.h_dim, out_features=self.h_dim),
                                         nn.LeakyReLU()
                                         )
        self.self_attn_V = nn.Sequential(nn.Linear(in_features=self.h_dim, out_features=self.h_dim),
                                         nn.LeakyReLU()
                                         )

    def forward(self, x):
        Q = self.self_attn_Q(x)
        K = self.self_attn_K(x)
        V = self.self_attn_V(x)

        return Q, K, V


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=input.device, dtype=input.dtype).view(1,
                                                                                                                     -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input


class CNN_ATTN(nn.Module):
    def __init__(self, configs):
        super(CNN_ATTN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)
        self.attn_network = attn_network(configs)
        self.sparse_max = Sparsemax(dim=-1)
        self.feat_len = configs.features_len

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        attentive_feat = self.calculate_attentive_feat(x_flat)
        return attentive_feat

    def self_attention(self, Q, K, scale=True, sparse=True, k=3):

        attention_weight = torch.bmm(Q.view(Q.shape[0], self.feat_len, -1), K.view(K.shape[0], -1, self.feat_len))

        attention_weight = torch.mean(attention_weight, dim=2, keepdim=True)

        if scale:
            d_k = torch.tensor(K.shape[-1]).float()
            attention_weight = attention_weight / torch.sqrt(d_k)
        if sparse:
            attention_weight_sparse = self.sparse_max(torch.reshape(attention_weight, [-1, self.feat_len]))
            attention_weight = torch.reshape(attention_weight_sparse, [-1, attention_weight.shape[1],
                                                                       attention_weight.shape[2]])
        else:
            attention_weight = self.softmax(attention_weight)

        return attention_weight

    def attention_fn(self, Q, K, scaled=False, sparse=True, k=1):

        attention_weight = torch.matmul(F.normalize(Q, p=2, dim=-1),
                                        F.normalize(K, p=2, dim=-1).view(K.shape[0], K.shape[1], -1, self.feat_len))

        if scaled:
            d_k = torch.tensor(K.shape[-1]).float()
            attention_weight = attention_weight / torch.sqrt(d_k)
            attention_weight = k * torch.log(torch.tensor(self.feat_len, dtype=torch.float32)) * attention_weight

        if sparse:
            attention_weight_sparse = self.sparse_max(torch.reshape(attention_weight, [-1, self.feat_len]))

            attention_weight = torch.reshape(attention_weight_sparse, attention_weight.shape)
        else:
            attention_weight = self.softmax(attention_weight)

        return attention_weight

    def calculate_attentive_feat(self, candidate_representation_xi):
        Q_xi, K_xi, V_xi = self.attn_network(candidate_representation_xi)
        intra_attention_weight_xi = self.self_attention(Q=Q_xi, K=K_xi, sparse=True)
        Z_i = torch.bmm(intra_attention_weight_xi.view(intra_attention_weight_xi.shape[0], 1, -1),
                        V_xi.view(V_xi.shape[0], self.feat_len, -1))
        final_feature = F.normalize(Z_i, dim=-1).view(Z_i.shape[0], -1)

        return final_feature
