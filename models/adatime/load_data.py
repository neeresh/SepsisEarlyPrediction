import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Load_Dataset(Dataset):
    def __init__(self, dataset, config):
        super().__init__()
        self.num_channels = config.d_channel

        x_data = dataset["samples"]
        y_data = dataset.get("labels").squeeze()
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)

        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)

        # # Check samples dimensions.
        # # The dimension of the data is expected to be (N, C, L)
        # # where N is the #samples, C: #channels, and L is the sequence length
        # if len(x_data.shape) == 2:
        #     x_data = x_data.unsqueeze(1)
        #
        # elif len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels:
        #     x_data = x_data.transpose(1, 2)

        # Normalize data
        if config.normalize:
            data_mean = torch.mean(x_data, dim=(0, 2))
            data_std = torch.std(x_data, dim=(0, 2))
            self.transform = transforms.Normalize(mean=data_mean, std=data_std)
        else:
            self.transform = None

        self.x_data = x_data.float()
        self.y_data = y_data.long() if y_data is not None else None
        self.len = x_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index]
        if self.transform:
            x = self.transform(self.x_data[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data[index].shape)
        y = self.y_data[index] if self.y_data is not None else None

        return x, y

    def __len__(self):
        return self.len


def data_generator(data_path, config, dtype):

    dataset_file = torch.load(data_path)
    dataset = Load_Dataset(dataset_file, config)

    if dtype == "test":
        shuffle = False
        drop_last = False

    else:
        shuffle = config.shuffle
        drop_last = config.drop_last

    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.batch_size,
                                              shuffle=shuffle, drop_last=drop_last, num_workers=4)

    return data_loader


def load_data(datapath, config, train_type='train'):
    dataloader = data_generator(datapath, config, dtype=train_type)

    return dataloader
