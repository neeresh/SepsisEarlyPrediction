import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from models.tfc.augmentations import DataTransform_FD, DataTransform_TD
import torch.fft as fft


def generate_freq(dataset, config):

    X_train = dataset["samples"]
    y_train = dataset['labels']

    data = list(zip(X_train, y_train))
    np.random.shuffle(data)

    X_train, y_train = zip(*data)
    X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0)

    if len(X_train.shape) < 3:
        X_train = X_train.unsqueeze(2)

    print(X_train.shape, y_train.shape)

    if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
        X_train = X_train.permute(0, 2, 1)

    X_train = X_train[:, :, :int(config.TSlength_aligned)]

    if isinstance(X_train, np.ndarray):
        x_data = torch.from_numpy(X_train)
    else:
        x_data = X_train

    """Transfer x_data to Frequency Domain. 
    If use fft.fft, the output has the same shape; if use fft.rfft, 
    the output shape is half of the time window."""

    x_data_f = fft.fft(x_data).abs()  # /(window_length) # rfft for real value inputs.

    return X_train, y_train, x_data_f


class Load_Dataset(Dataset):

    def __init__(self, dataset, config, training_mode):

        super(Load_Dataset, self).__init__()

        self.training_mode = training_mode
        X_train = dataset["samples"]
        y_train = dataset["labels"]

        data = list(zip(X_train, y_train))
        np.random.shuffle(data)

        X_train, y_train = zip(*data)
        X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0).squeeze()

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        # if X_train.shape.index(min(X_train.shape)) != 1: # make sure the Channels in second dim
        #     X_train = X_train.permute(0, 2, 1)

        """Align the TS length between source and target datasets"""
        X_train = X_train[:, :, :int(config.TSlength_aligned)]

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        """Transfer x_data to Frequency Domain. If use fft.fft, the output has the same shape; if use fft.rfft, 
        the output shape is half of the time window."""
        window_length = self.x_data.shape[-1]
        self.x_data_f = fft.fft(self.x_data).abs()
        self.len = X_train.shape[0]

        """Augmentation"""
        if training_mode == "pre_train":  # no need to apply Augmentations in other modes
            self.aug1 = DataTransform_TD(self.x_data, config)
            self.aug1_f = DataTransform_FD(self.x_data_f, config)  # [7360, 1, 90]

    def __getitem__(self, index):
        if self.training_mode == "pre_train":
            # data, labels, aug1, data_f, aug1_f
            return self.x_data[index], self.y_data[index], self.aug1[index], \
                self.x_data_f[index], self.aug1_f[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], \
                self.x_data_f[index], self.x_data_f[index]

    def __len__(self):
        return self.len


def data_generator(dataset_pt, configs):

    dataset = Load_Dataset(dataset_pt, configs, training_mode='pre_train')
    loader = DataLoader(dataset=dataset, batch_size=configs.batch_size, shuffle=True,
                              drop_last=configs.drop_last, num_workers=4)

    return loader
