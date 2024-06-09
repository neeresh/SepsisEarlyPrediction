import random

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset, Sampler, DataLoader


class BatchRandomSampler(Sampler):
    """
    Batches the data consecutively and randomly samples
    by batch without replacement.
    """

    def __init__(self, data_source, batch_size):
        super(BatchRandomSampler, self).__init__(data_source)
        it_end = len(data_source) - batch_size + 1
        self.batches = [range(i, i + batch_size) for i in range(0, it_end, batch_size)]
        self.data_source = data_source

    def __iter__(self):
        random.shuffle(self.batches)
        return (i for b in self.batches for i in b)

    def __len__(self):
        return len(self.data_source)


class Dataset(Dataset):
    def __init__(self, examples, lengths_list, bucket_diff=4):
        max_len = max(lengths_list)  # 336
        num_buckets = max_len // bucket_diff  # 84

        buckets = [[] for _ in range(num_buckets)]

        # Loading data into buckets according to the patient's data size
        for example in tqdm.tqdm(examples, desc='Binning and sorting the data according to patients record length'):
            bid = min(len(example) // bucket_diff, num_buckets - 1)  # 83
            buckets[bid].append(example)

        sort_fn = lambda x: len(x)
        for b in buckets:
            b.sort(key=sort_fn)

        data = [d for b in buckets for d in b]
        self.data = data

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        inputs = self.data[idx].drop(['SepsisLabel', 'ICULOS'], axis=1).values
        targets = self.data[idx].SepsisLabel.values

        return inputs, targets


class TimeSeriesDataset(Dataset):
    def __init__(self, examples, window_size, step_size):
        self.windows = []
        self.labels = []
        self.window_size = window_size
        self.step_size = step_size

        for example in tqdm.tqdm(examples, desc="Windowing the data"):
            self._create_windows(example)

    def _create_windows(self, example):
        num_windows = (len(example) - self.window_size) // self.step_size + 1
        for i in range(num_windows):
            start = i * self.step_size
            end = start + self.window_size
            window = example.iloc[start:end]
            self.windows.append(window.drop(['SepsisLabel', 'ICULOS'], axis=1).values)
            self.labels.append(window['SepsisLabel'].values)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


def collate(inputs, targets):
    max_t = max(inp.shape[0] for inp in inputs)
    x_shape = (len(inputs), max_t, inputs[0].shape[1])
    y_shape = (len(inputs), max_t)
    x = np.zeros(x_shape, dtype=np.float32)
    y = np.zeros(y_shape, dtype=np.float32)
    for e, (inp, tar) in enumerate(zip(inputs, targets)):
        x[e, :inp.shape[0], :] = inp
        y[e, :inp.shape[0]] = tar

    # y = torch.tensor(np.concatenate(y)).type(torch.LongTensor)
    y = torch.tensor(np.concatenate(y)).to('cuda')
    x = torch.tensor(x).to('cuda')

    return x, y


def make_raw_loader(examples, lengths_list, batch_size, is_sepsis, num_workers=8,
                    use_sampler=False, window_size=None, step_size=None):

    if window_size is not None:
        dataset = TimeSeriesDataset(examples, window_size, step_size)
    else:
        dataset = Dataset(examples=examples, lengths_list=lengths_list)

    if use_sampler:
        sampler = BatchRandomSampler(dataset, batch_size)
    else:
        sampler = None

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                        collate_fn=lambda batch: zip(*batch), drop_last=True)

    return loader
