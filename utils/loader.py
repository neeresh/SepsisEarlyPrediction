import logging
import random

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset, Sampler, DataLoader


class DatasetWithPadding(Dataset):
    def __init__(self, training_examples_list, lengths_list, is_sepsis):
        self.data, self.labels = self._create_dataset(training_examples_list, lengths_list, is_sepsis)

    def _create_dataset(self, training_examples_list, lengths_list, is_sepsis):
        logging.info(f"Input features ({len(training_examples_list[0].columns)}): {training_examples_list[0].columns}")
        data, labels = [], []
        max_time_step = max(lengths_list)
        for patient_data, sepsis in zip(training_examples_list, is_sepsis):
            pad = (max_time_step - len(patient_data))
            patient_data = np.pad(patient_data, ((0, pad), (0, 0)), mode='constant')
            data.append(patient_data)
            labels.append(sepsis)

        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


class DatasetWithWindows(Dataset):
    def __init__(self, training_examples_list, lengths_list, is_sepsis, window_size=6, step_size=6):
        self.window_size = window_size
        self.step_size = step_size
        self.data, self.labels = self._create_dataset(training_examples_list, lengths_list, is_sepsis,
                                                      window_size, step_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def _create_dataset(self, training_examples_list, lengths_list, is_sepsis, window_size, step_size):
        logging.info(f"Input features ({len(training_examples_list[0].columns)}): {training_examples_list[0].columns}")
        x, y = [], []
        for time_step in range(window_size, len(training_examples_list) - step_size):
            window = training_examples_list[time_step - window_size : time_step]
            label = int(np.array(is_sepsis)[time_step : time_step + step_size].max())
            x.append(window), y.append(label)

        return np.array(x), np.array(y)


def make_loader(examples, lengths_list, is_sepsis, batch_size, num_workers=8, mode="window"):

    if mode == "window":
        dataset = DatasetWithWindows(training_examples_list=examples, lengths_list=lengths_list, is_sepsis=is_sepsis,
                                     window_size=6, step_size=5)
        logging.info(f"Window size: {dataset.window_size} & Step size: {dataset.step_size}")

    elif mode == "padding":
        dataset = DatasetWithPadding(training_examples_list=examples, lengths_list=lengths_list, is_sepsis=is_sepsis)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    logging.info(f"Num of training examples: {len(train_dataset)}")
    logging.info(f"Num of test examples: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
