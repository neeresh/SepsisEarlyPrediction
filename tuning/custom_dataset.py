import os

import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils.path_utils import project_root


class DatasetWithPadding(Dataset):
    def __init__(self, training_examples_list, lengths_list, is_sepsis):
        self.data, self.labels = self._create_dataset(training_examples_list, lengths_list, is_sepsis)

    def _create_dataset(self, training_examples_list, lengths_list, is_sepsis):
        data, labels = [], []
        max_time_step = 336
        for patient_data, sepsis in zip(training_examples_list, is_sepsis):
            patient_data = patient_data.drop(['PatientID', 'SepsisLabel'], axis=1)
            pad = (max_time_step - len(patient_data), 0)
            patient_data = np.pad(patient_data, pad_width=((0, pad[0]), (0, 0)), mode='constant').astype(np.float32)

            data.append(torch.from_numpy(patient_data))
            labels.append(sepsis)

        return data, labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item], self.labels[item]


def get_train_val_test_indices():
    # is_sepsis_file = pd.read_csv(os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt'))
    is_sepsis_file = pd.read_csv('./data/processed/is_sepsis.txt')

    train_temp, test = train_test_split(is_sepsis_file, test_size=0.2, random_state=42)
    train, val = train_test_split(train_temp, test_size=0.2, random_state=42)

    train_indices = train.index.values
    val_indices = val.index.values
    test_indices = test.index.values

    return train_indices, val_indices, test_indices


def initialize_experiment():
    data_file = "final_dataset.pickle"

    # [[patient1], [patient2], [patient3], ..., [patientN]]
    # training_examples = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', data_file))
    training_examples = pd.read_pickle(f"./data/processed/{data_file}")

    # with open(os.path.join(project_root(), 'data', 'processed', 'lengths.txt')) as f:
    #     lengths_list = [int(length) for length in f.read().splitlines()]
    # with open(os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt')) as f:
    #     is_sepsis = [int(is_sep) for is_sep in f.read().splitlines()]

    with open(os.path.join('./data/processed/lengths.txt')) as f:
        lengths_list = [int(length) for length in f.read().splitlines()]
    with open(os.path.join('./data/processed/is_sepsis.txt')) as f:
        is_sepsis = [int(is_sep) for is_sep in f.read().splitlines()]

    return training_examples, lengths_list, is_sepsis


def get_starters():
    """
    This is the first method to be called.
    """
    train_indicies, val_indices, test_indicies = get_train_val_test_indices()
    examples, lengths_list, is_sepsis = initialize_experiment()

    return train_indicies, val_indices, test_indicies, examples, lengths_list, is_sepsis


def load_data(train_indicies, val_indices, test_indicies, examples, lengths_list, is_sepsis):

    train_samples = [examples[idx] for idx in train_indicies]
    val_samples = [examples[idx] for idx in val_indices]
    test_samples = [examples[idx] for idx in test_indicies]

    train_lengths_list = [lengths_list[idx] for idx in train_indicies]
    val_lengths_list = [lengths_list[idx] for idx in val_indices]
    test_lengths_list = [lengths_list[idx] for idx in test_indicies]

    is_sepsis_train = [is_sepsis[idx] for idx in train_indicies]
    is_sepsis_val = [is_sepsis[idx] for idx in val_indices]
    is_sepsis_test = [is_sepsis[idx] for idx in test_indicies]

    train_dataset = DatasetWithPadding(training_examples_list=train_samples, lengths_list=train_lengths_list,
                                       is_sepsis=is_sepsis_train)
    val_dataset = DatasetWithPadding(training_examples_list=val_samples, lengths_list=val_lengths_list,
                                     is_sepsis=is_sepsis_val)
    test_dataset = DatasetWithPadding(training_examples_list=test_samples, lengths_list=test_lengths_list,
                                      is_sepsis=is_sepsis_test)

    return train_dataset, val_dataset, test_dataset
