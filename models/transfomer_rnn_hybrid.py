import copy
import math
import torch
import torch.nn as nn
import numpy as np
import time
import tqdm
from sklearn.model_selection import StratifiedKFold

from train_model import TrainModel
from utils.loader import make_loader, collate
from utils.normalized_utility_score import normalized_utility_score


def _create_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask) == 0
    if torch.cuda.is_available():
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = nn.functional.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class MultiHeadSelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_of_heads=8, dropout=0.1):
        super(MultiHeadSelfAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.d_k = hidden_size // num_of_heads
        self.h = num_of_heads
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.hidden_size)
        output = self.out(concat)
        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_layers, hidden_size, num_of_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_layers = num_layers
        self.layers = get_clones(MultiHeadSelfAttentionCell(hidden_size=hidden_size,
                                                            num_of_heads=num_of_heads,
                                                            dropout=dropout), n=num_layers)

    def forward(self, x, mask=None):
        for i in range(self.num_layers):
            x = self.layers[i](x, mask=mask)
        return x


class MultiHeadSelfAttentionCell(nn.Module):
    def __init__(self, hidden_size, num_of_heads=8, dropout=0.1):
        super(MultiHeadSelfAttentionCell, self).__init__()
        self.self_attention = MultiHeadSelfAttentionLayer(hidden_size=hidden_size,
                                                          num_of_heads=num_of_heads,
                                                          dropout=dropout)
        self.norm = Norm(hidden_size=hidden_size)

    def forward(self, x, mask=None):
        out = self.self_attention(x, x, x, mask=mask)
        out = self.norm(x + out)
        return out


class Norm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(Norm, self).__init__()
        self.hidden_size = hidden_size
        self.alpha = nn.Parameter(torch.ones(self.hidden_size))
        self.bias = nn.Parameter(torch.zeros(self.hidden_size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class PositionalEncoder(nn.Module):
    def __init__(self, hidden_size, max_seq_len=1000):
        super(PositionalEncoder, self).__init__()
        self.hidden_size = hidden_size
        pe = _create_position_encoding(hidden_size=hidden_size, max_seq_len=max_seq_len)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.hidden_size)
        seq_len = x.size(1)
        if torch.cuda.is_available():
            x = x.cuda()
        x = x + self.pe[:, :seq_len]
        return x


def _create_position_encoding(hidden_size, max_seq_len=1000):
    pe = torch.zeros(max_seq_len, hidden_size)
    last_iter = hidden_size if hidden_size % 2 == 0 else hidden_size - 1
    for pos in range(max_seq_len):
        for i in range(0, last_iter, 2):
            pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / hidden_size)))
            pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / hidden_size)))
    pe = pe.unsqueeze(0)
    return pe


class PositionalEncoderConcat(nn.Module):
    def __init__(self, hidden_size, max_seq_len=1000):
        super(PositionalEncoderConcat, self).__init__()
        self.hidden_size = hidden_size
        pe = _create_position_encoding(hidden_size=hidden_size, max_seq_len=max_seq_len)
        self.register_buffer('pe', pe)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = torch.cat((x, self.pe[:, :seq_len].repeat(batch_size, 1, 1)), 2)
        if torch.cuda.is_available():
            x = x.cuda()
        return x


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_of_heads, num_layers, dropout, eps=1e-6, to_concat=True):
        super(GRUClassifier, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, num_layers=num_layers, hidden_size=hidden_size, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        rnn_out, hidden = self.rnn(x)
        out = self.linear(rnn_out)
        return out


class SelfAttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_of_heads, num_layers, dropout, eps=1e-6, to_concat=True):
        super(SelfAttentionClassifier, self).__init__()
        if to_concat:
            self.positional_encoding = PositionalEncoderConcat(hidden_size)
            model_hidden_size = hidden_size * 2
        else:
            self.positional_encoding = PositionalEncoder(hidden_size)
            model_hidden_size = hidden_size
        self.self_attention = MultiHeadSelfAttention(num_layers=num_layers,
                                                     hidden_size=model_hidden_size, num_of_heads=num_of_heads,
                                                     dropout=dropout)
        self.linear = nn.Linear(model_hidden_size, 1)

    def forward(self, x):
        x = self.positional_encoding(x)
        mask = _create_mask(x.shape[1])
        self_attention_out = self.self_attention(x, mask=mask)
        out = self.linear(self_attention_out)
        return out


class TransformerRNNHybrid:
    def __init__(self, config, writer=None, eval_set=None):
        self.conf = config
        self.model = SelfAttentionClassifier(input_size=config['input_size'],
                                             hidden_size=config['hidden_size'],
                                             num_of_heads=config['num_of_heads'],
                                             num_layers=config['num_layers'],
                                             dropout=config['dropout'],
                                             to_concat=config['to_concat'])
        # self.criterion = torch.nn.BCEWithLogitsLoss(size_average=self.conf['size_average'])
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf['lr'])
        self.writer = writer
        self.eval_set = eval_set

    def fit(self, examples, lengths_list, is_sepsis):
        loader = make_loader(examples, lengths_list=lengths_list,
                                 is_sepsis=is_sepsis, batch_size=self.conf['batch_size'])

        abs_i = 0
        for epoch_i in range(self.conf['epochs_num']):
            self.model.train(mode=True)
            start = time.time()
            total_tokens = 0
            total_loss = 0
            tokens = 0
            rows_per_sec = 0

            tq = tqdm.tqdm(loader)
            for i, (inputs, targets) in enumerate(tq):
                self.optimizer.zero_grad()
                x, y = collate(inputs, targets)
                print(x.shape, y.shape)
                output = self.model(x)
                batch_size, _, out_dim = output.size()
                output = output.view(-1)

                # weight = torch.tensor(np.ones((output.size(0), ))*10, requires_grad=False)
                loss = self.criterion(output, y)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.conf['clipping'])
                self.optimizer.step()
                loss_val = loss.item() / batch_size

                total_loss += loss_val
                total_tokens += x.shape[0]
                tokens += x.shape[0]
                if i % 50 == 1:
                    elapsed = time.time() - start
                    rows_per_sec = tokens / elapsed
                    start = time.time()
                    tokens = 0
                if self.writer is not None:
                    self.writer.add_scalar('train_loss', loss_val, abs_i)
                    self.writer.add_scalar('grad_norm', grad_norm, abs_i)
                    if i % 10 == 0:
                        # for name, param in self.model.named_parameters():
                            # self.writer.add_histogram(name, param.clone().cpu().detach().numpy(), abs_i, bins='doane')
                        pass

                tq.set_postfix(iter=i, loss=loss_val, rows_per_sec=rows_per_sec)
                abs_i += 1
            if self.eval_set is not None:
                y_pred_eval, y_ref_eval = self.predict(self.eval_set[0][0])
                nus, _, f_score = normalized_utility_score(targets=y_ref_eval, predictions=y_pred_eval)
                self.writer.add_scalar('eval_norm_utility_score', nus, abs_i)
                self.writer.add_scalar('eval_f_score', f_score, abs_i)
                print(nus)

    def predict(self, examples):
        self.model.train(mode=False)
        predictions = []
        references = []
        probas = []
        with torch.no_grad():
            for example in tqdm.tqdm(examples):
                reference = example['SepsisLabel'].values
                example = torch.tensor([example.drop(['SepsisLabel', 'ICULOS'], axis=1).values.astype(np.float32)])

                proba = self.model(example)
                proba = nn.functional.sigmoid(proba)
                proba = np.concatenate(proba.data.detach().cpu().numpy()[0])
                prediction = np.where(proba > 0.5, 1, 0)

                predictions.append(prediction)
                probas.append(proba)
                references.append(reference)

        return predictions, references

def get_train_test_splits(ind_train, ind_test, training_examples, lengths_list, is_sepsis):
    x_train = [t for i, t in enumerate(training_examples) if i in ind_train]
    x_train_lens = [t for i, t in enumerate(lengths_list) if i in ind_train]
    is_sepsis_train = [t for i, t in enumerate(is_sepsis) if i in ind_train]

    x_test = [t for i, t in enumerate(training_examples) if i in ind_test]
    x_test_lens = [t for i, t in enumerate(lengths_list) if i in ind_test]
    is_sepsis_test = [t for i, t in enumerate(is_sepsis) if i in ind_test]

    # print(len(x_train), len(x_train_lens), len(is_sepsis_train), len(x_test), len(x_test), len(is_sepsis_test))

    return x_train, x_train_lens, is_sepsis_train, x_test, x_test_lens, is_sepsis_test


# if __name__ == '__main__':
#     from utils.config import transformer_rnn_param
#     train_model = TrainModel()
#     training_examples, lengths_list, is_sepsis, writer, destination_path = train_model.initialize_experiment()
#
#     skf = StratifiedKFold(n_splits=5)
#
#     model = TransformerRNNHybrid(config=transformer_rnn_param, writer=writer)
#     model.model.to('cuda')
#
#     for i, (ind_train, ind_test) in enumerate(skf.split(training_examples, is_sepsis)):
#         # Getting splits
#         x_train, x_train_lens, is_sepsis_train, x_test, x_test_lens, is_sepsis_test = get_train_test_splits(
#             ind_train, ind_test, training_examples, lengths_list, is_sepsis
#         )
#
#         model.eval_set = [(x_test, is_sepsis_test), (x_train, is_sepsis_train)]
#         model.fit(examples=x_train, lengths_list=x_train_lens, is_sepsis=is_sepsis_train)

if __name__ == '__main__':
    from utils.config import transformer_rnn_param
    train_model = TrainModel()
    training_examples, lengths_list, is_sepsis, writer, destination_path = train_model.initialize_experiment()

    model = TransformerRNNHybrid(transformer_rnn_param)
    model.model.to('cuda')

    print(model.model)
    # model.fit(examples=training_examples, lengths_list=lengths_list, is_sepsis=is_sepsis)
