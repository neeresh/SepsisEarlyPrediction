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


def make_loader(examples, lengths_list, is_sepsis, batch_size, num_workers=8, use_sampler=False):
    # dataset = Dataset(examples=examples, lengths_list=lengths_list)
    dataset = DatasetWithPadding(training_examples_list=examples, lengths_list=lengths_list, is_sepsis=is_sepsis)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    logging.info(f"Num of training examples: {len(train_dataset)}")
    logging.info(f"Num of test examples: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
