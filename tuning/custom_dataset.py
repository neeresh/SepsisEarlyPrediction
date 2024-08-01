import os

import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils.path_utils import project_root


# from utils.loader import DatasetWithPadding, DatasetWithPaddingMasking


class DatasetWithPaddingMasking(Dataset):
    def __init__(self, training_examples_list, lengths_list, is_sepsis):
        self.data, self.labels, self.mask = self._create_dataset(training_examples_list, lengths_list, is_sepsis)

    def _create_dataset(self, training_examples_list, lengths_list, is_sepsis):
        data, labels, masks = [], [], []

        max_time_step = 336  # The maximum sequence length for padding

        for patient_data, length, sepsis in zip(training_examples_list, lengths_list, is_sepsis):

            patient_data = patient_data.drop(['PatientID', 'SepsisLabel'], axis=1)
            original_length = len(patient_data)
            padding_length = max_time_step - original_length
            patient_data = np.pad(patient_data, pad_width=((0, padding_length), (0, 0)), mode='constant').astype(
                np.float32)

            # mask = 1 for valid entries, 0 for padding
            mask = np.ones((max_time_step,), dtype=bool)
            if padding_length > 0:
                mask[original_length:] = False

            # Convert to tensors
            data.append(torch.from_numpy(patient_data))
            labels.append(sepsis)
            masks.append(torch.from_numpy(mask))  # Expand mask to match input shape (max_time_step, 1)

        return data, labels, masks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item], self.mask[item]


def get_train_val_test_indices(majority_class, include_val):
    file_path = os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt')
    sepsis = pd.Series(open(file_path, 'r').read().splitlines()).astype(int)

    assert len(sepsis) == 40336, f"Total number of patients should be 40036 but given {len(sepsis)}"

    positive_sepsis_idxs = sepsis[sepsis == 1].index
    negative_sepsis_idxs = sepsis[sepsis == 0].sample(frac=majority_class, random_state=42).index
    all_samples = list(positive_sepsis_idxs) + list(negative_sepsis_idxs)
    np.random.shuffle(all_samples)

    train_indicies, temp_indicies = train_test_split(all_samples, test_size=0.2, random_state=42)  # 80 20
    val_indicies, test_indicies = train_test_split(temp_indicies, test_size=0.5, random_state=42)  # 10 10

    if include_val:
        return train_indicies, val_indicies, test_indicies

    else:
        return train_indicies, None, test_indicies


def get_data(data_file):
    training_examples = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', data_file))

    with open(os.path.join(project_root(), 'data', 'processed', 'lengths.txt')) as f:
        lengths_list = [int(length) for length in f.read().splitlines()]

    with open(os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt')) as f:
        is_sepsis = [int(is_sep) for is_sep in f.read().splitlines()]

    return training_examples, lengths_list, is_sepsis


def get_starters(majority_class, data_file, include_val=False):
    train_indices, val_indices, test_indices = get_train_val_test_indices(majority_class, include_val)
    examples, lengths_list, is_sepsis = get_data(data_file)

    return train_indices, val_indices, test_indices, examples, lengths_list, is_sepsis


def load_data(train_indices, val_indices, test_indices, examples, lengths_list, is_sepsis, batch_size):
    train_samples = [examples[idx] for idx in train_indices]
    test_samples = [examples[idx] for idx in test_indices]

    train_lengths_list = [lengths_list[idx] for idx in train_indices]
    test_lengths_list = [lengths_list[idx] for idx in test_indices]

    is_sepsis_train = [is_sepsis[idx] for idx in train_indices]
    is_sepsis_test = [is_sepsis[idx] for idx in test_indices]

    train_dataset = DatasetWithPaddingMasking(training_examples_list=train_samples, lengths_list=train_lengths_list,
                                              is_sepsis=is_sepsis_train)
    test_dataset = DatasetWithPaddingMasking(training_examples_list=test_samples, lengths_list=test_lengths_list,
                                             is_sepsis=is_sepsis_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=int(batch_size), shuffle=False, num_workers=4)

    if val_indices:
        val_samples = [examples[idx] for idx in val_indices]
        val_lengths_list = [lengths_list[idx] for idx in val_indices]
        is_sepsis_val = [is_sepsis[idx] for idx in val_indices]

        val_dataset = DatasetWithPaddingMasking(training_examples_list=val_samples, lengths_list=val_lengths_list,
                                                is_sepsis=is_sepsis_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(batch_size), shuffle=False,
                                                 num_workers=4)

        return train_loader, val_loader, test_loader

    return train_loader, None, test_loader
