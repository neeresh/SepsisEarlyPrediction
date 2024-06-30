import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler, DataLoader

from utils.path_utils import project_root
from torch.nn.utils.rnn import pad_sequence
import torch


def collate_fn(batch):
    # Sequeneces and Lengths
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])

    # Padding
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)

    # Creating masks
    masks = torch.zeros(sequences_padded.size(0), sequences_padded.size(1), dtype=torch.bool)
    for i, length in enumerate(lengths):
        masks[i, :length] = 1

    return sequences_padded, masks, torch.tensor(labels, dtype=torch.long)


class DatasetWithPadding(Dataset):
    def __init__(self, training_examples_list, lengths_list, is_sepsis):
        self.data, self.labels = self._create_dataset(training_examples_list, lengths_list, is_sepsis)

    def _create_dataset(self, training_examples_list, lengths_list, is_sepsis):
        logging.info(f"Input features ({len(training_examples_list[0].columns)}): {training_examples_list[0].columns}")
        data, labels = [], []
        # max_time_step = max(lengths_list)
        max_time_step = 336
        for patient_data, sepsis in tqdm.tqdm(zip(training_examples_list, is_sepsis), desc="Padding...",
                                              total=len(training_examples_list)):
            patient_data = patient_data.drop(['PatientID', 'SepsisLabel'], axis=1)
            pad = (max_time_step - len(patient_data), 0)
            patient_data = np.pad(patient_data, pad_width=((0, pad[0]), (0, 0)), mode='constant').astype(np.float32)

            data.append(torch.from_numpy(patient_data))
            labels.append(sepsis)

        logging.info(f"Total number of samples after applying window method: ({len(data)})")
        logging.info(f"Distribution of Sepsis:\n{pd.Series(labels).value_counts()}")

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

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def _create_dataset(self, training_examples_list, lengths_list, is_sepsis, window_size, step_size):
        x, y = [], []

        patients_counter = 0
        for idx, subset in tqdm.tqdm(enumerate(training_examples_list), desc="Creating windows",
                                     total=len(training_examples_list)):

            sepsis = subset['SepsisLabel'].values.astype(np.int64)
            patient_data = subset.drop(['SepsisLabel', 'PatientID'], axis=1).values.astype(np.float32)

            patients_counter += 1
            for idx, time_step in enumerate(range(window_size, len(patient_data) - step_size)):
                window = patient_data[time_step - window_size: time_step]
                label = int(sepsis[time_step: time_step + step_size].max())
                x.append(window)
                y.append(label)

        x, y = np.array(x), np.array(y)

        logging.info(f"Input features ({len(subset.drop(['SepsisLabel', 'PatientID'], axis=1).columns)}): "
                     f"{subset.drop(['SepsisLabel', 'PatientID'], axis=1).columns}")
        logging.info(f"Total number of samples after applying window method: ({len(x)})")
        logging.info(f"Distribution of sSepsis:\n{pd.Series(y).value_counts()}")
        logging.info(f"Shape of the data: {x.shape}")
        logging.info(f"Total number of patients: {patients_counter}")

        return x, y


class DefaultDataset(Dataset):
    def __init__(self, training_examples_list, lengths_list, is_sepsis):
        self.data, self.labels = self._create_dataset(training_examples_list, lengths_list, is_sepsis)

    def _create_dataset(self, training_examples_list, lengths_list, is_sepsis):
        logging.info(f"Input features ({len(training_examples_list[0].columns)}): {training_examples_list[0].columns}")
        data, labels = [], []
        max_time_step = max(lengths_list)
        for patient_data, sepsis in tqdm.tqdm(zip(training_examples_list, is_sepsis), desc="Retrieving dataset...",
                                              total=len(training_examples_list)):
            patient_data = patient_data.drop(['PatientID', 'SepsisLabel'], axis=1)
            data.append(torch.from_numpy(patient_data.values.astype(np.float32)))
            labels.append(sepsis)

        logging.info(f"Total number of samples after applying window method: ({len(data)})")
        logging.info(f"Distribution of Sepsis:\n{pd.Series(labels).value_counts()}")

        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


class DatasetWithPaddingMasking(Dataset):
    def __init__(self, training_examples_list, lengths_list, is_sepsis):
        self.data, self.labels, self.mask = self._create_dataset(training_examples_list, lengths_list, is_sepsis)

    def _create_dataset(self, training_examples_list, lengths_list, is_sepsis):
        logging.info(f"Input features ({len(training_examples_list[0].columns)}): {training_examples_list[0].columns}")
        data, labels, masks = [], [], []
        # max_time_step = max(lengths_list)
        max_time_step = 336
        for patient_data, sepsis in tqdm.tqdm(zip(training_examples_list, is_sepsis), desc="Padding...",
                                              total=len(training_examples_list)):
            patient_data = patient_data.drop(['PatientID', 'SepsisLabel'], axis=1)
            pad = (max_time_step - len(patient_data), 0)
            patient_data = np.pad(patient_data, pad_width=((0, pad[0]), (0, 0)), mode='constant').astype(np.float32)

            # Creating mask
            # Padding real data with True and padded data with False
            mask = np.ones_like(patient_data)
            mask[pad[0]:, :] = 0  # Mask the padded elements

            data.append(torch.from_numpy(patient_data))
            labels.append(sepsis)
            masks.append(torch.from_numpy(mask.astype('bool')))

        logging.info(f"Total number of samples after applying window method: ({len(data)})")
        logging.info(f"Distribution of Sepsis:\n{pd.Series(labels).value_counts()}")

        return data, labels, masks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item], self.mask[item]


def get_train_test_indicies():
    is_sepsis_file = pd.read_csv(os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt'), header=None)
    assert len(is_sepsis_file) == 40336, f"is_sepsis.txt didn't load properly"

    train, test = train_test_split(is_sepsis_file, test_size=0.2, random_state=42)

    train_indicies = train.index.values
    test_indicies = test.index.values

    return train_indicies, test_indicies


def get_train_val_test_indices():
    is_sepsis_file = pd.read_csv(os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt'), header=None)
    assert len(is_sepsis_file) == 40336, f"Check the input and output size"
    train_temp, test = train_test_split(is_sepsis_file, test_size=0.2, random_state=42)

    train, val = train_test_split(train_temp, test_size=0.2, random_state=42)

    train_indices = train.index.values
    val_indices = val.index.values
    test_indices = test.index.values

    return train_indices, val_indices, test_indices


def make_loader(examples, lengths_list, is_sepsis, batch_size, mode, num_workers=8,
                train_indicies=None, test_indicies=None):

    if train_indicies is None and test_indicies is None:
        print("Loading from defined indicies")
        train_indicies, test_indicies = get_train_test_indicies()

    train_samples = [examples[idx] for idx in train_indicies]
    test_samples = [examples[idx] for idx in test_indicies]

    train_lengths_list = [lengths_list[idx] for idx in train_indicies]
    test_lengths_list = [lengths_list[idx] for idx in test_indicies]

    is_sepsis_train = [is_sepsis[idx] for idx in train_indicies]
    is_sepsis_test = [is_sepsis[idx] for idx in test_indicies]

    if mode == "window":
        train_dataset = DatasetWithWindows(training_examples_list=train_samples, lengths_list=train_lengths_list,
                                           is_sepsis=is_sepsis_train, window_size=8, step_size=6)
        test_dataset = DatasetWithWindows(training_examples_list=test_samples, lengths_list=test_lengths_list,
                                          is_sepsis=is_sepsis_test, window_size=8, step_size=6)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        logging.info(f"Window size: {train_dataset.window_size} & Step size: {train_dataset.step_size}")

    elif mode == "padding":
        train_dataset = DatasetWithPadding(training_examples_list=train_samples, lengths_list=train_lengths_list,
                                           is_sepsis=is_sepsis_train)
        test_dataset = DatasetWithPadding(training_examples_list=test_samples, lengths_list=test_lengths_list,
                                          is_sepsis=is_sepsis_test, )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif mode == "default":
        train_dataset = DefaultDataset(training_examples_list=train_samples, lengths_list=train_lengths_list,
                                       is_sepsis=is_sepsis_train)
        test_dataset = DefaultDataset(training_examples_list=test_samples, lengths_list=test_lengths_list,
                                      is_sepsis=is_sepsis_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 collate_fn=collate_fn)

    elif mode == "padding_masking":
        train_dataset = DatasetWithPaddingMasking(training_examples_list=train_samples, lengths_list=train_lengths_list,
                                                  is_sepsis=is_sepsis_train)
        test_dataset = DatasetWithPaddingMasking(training_examples_list=test_samples, lengths_list=test_lengths_list,
                                                 is_sepsis=is_sepsis_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logging.info(f"Num of training examples: {len(train_dataset)}")
    logging.info(f"Num of test examples: {len(test_dataset)}")

    return train_loader, test_loader, train_indicies, test_indicies


def initialize_experiment(data_file=None):
    if data_file is not None:
        data_file = "training_ffill_bfill_zeros.pickle"

    data_file = "final_dataset.pickle"

    print(f"Dataset used: {data_file}")

    # [[patient1], [patient2], [patient3], ..., [patientN]]
    training_examples = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', data_file))

    with open(os.path.join(project_root(), 'data', 'processed', 'lengths.txt')) as f:
        lengths_list = [int(length) for length in f.read().splitlines()]
    with open(os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt')) as f:
        is_sepsis = [int(is_sep) for is_sep in f.read().splitlines()]

    return training_examples, lengths_list, is_sepsis


if __name__ == '__main__':
    training_examples, lengths_list, is_sepsis = initialize_experiment()
    train_loader, test_loader = make_loader(training_examples, lengths_list, is_sepsis, batch_size=128)

    for idx, patient_data in enumerate(train_loader):
        if idx == 1:
            print(patient_data)
            break
