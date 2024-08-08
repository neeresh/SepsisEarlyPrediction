import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler, DataLoader

from utils.helpers import get_features
from utils.path_utils import project_root
from torch.nn.utils.rnn import pad_sequence
import torch

lgbm_features = ['Temp', 'SBP', 'EtCO2', 'FiO2', 'pH', 'PaCO2', 'BUN', 'Alkalinephos', 'Calcium',
                 'Chloride', 'Creatinine', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
                 'Bilirubin_total', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'Age',
                 'HospAdmTime', 'ICULOS', 'interval_f1_O2Sat', 'interval_f1_Temp', 'interval_f2_SBP',
                 'interval_f1_DBP', 'interval_f2_DBP', 'interval_f1_Resp', 'interval_f2_Resp',
                 'interval_f1_EtCO2', 'interval_f1_HCO3', 'interval_f1_FiO2', 'interval_f2_FiO2',
                 'interval_f1_PaCO2', 'interval_f2_PaCO2', 'interval_f2_SaO2', 'interval_f2_AST',
                 'diff_f_BUN', 'interval_f1_Calcium', 'interval_f2_Calcium',
                 'diff_f_Calcium', 'diff_f_Creatinine', 'interval_f1_Glucose',
                 'diff_f_Glucose', 'interval_f1_Lactate', 'interval_f2_Lactate',
                 'diff_f_Magnesium', 'interval_f1_Phosphate', 'diff_f_Phosphate',
                 'interval_f1_Potassium', 'interval_f1_Bilirubin_total',
                 'interval_f1_Hct', 'diff_f_Hct', 'diff_f_Hgb', 'interval_f2_PTT',
                 'diff_f_PTT', 'diff_f_WBC', 'diff_f_Platelets', 'HR_max', 'HR_mean',
                 'HR_std', 'O2Sat_max', 'O2Sat_mean', 'SBP_mean', 'MAP_max',
                 't_suspicion']  # Total: 70


def collate_fn(batch):
    # Sequences and Lengths
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


# class DatasetWithPaddingMasking(Dataset):
#     def __init__(self, training_examples_list, lengths_list, is_sepsis):
#         self.data, self.labels, self.mask = self._create_dataset(training_examples_list, lengths_list, is_sepsis)
#
#     def _create_dataset(self, training_examples_list, lengths_list, is_sepsis):
#
#         logging.info(f"Input features ({len(training_examples_list[0].columns)}): {training_examples_list[0].columns}")
#         data, labels, masks = [], [], []
#
#         # max_time_step = max(lengths_list)
#         max_time_step = 336
#         for patient_data, sepsis in tqdm.tqdm(zip(training_examples_list, is_sepsis), desc="Padding...",
#                                               total=len(training_examples_list)):
#             patient_data = patient_data.drop(['PatientID', 'SepsisLabel'], axis=1)
#             original_length = len(patient_data)
#             pad = (max_time_step - original_length, 0)
#             patient_data = np.pad(patient_data, pad_width=((0, pad[0]), (0, 0)), mode='constant').astype(np.float32)
#
#             # Creating mask
#             # Padding real data with True and padded data with False
#             mask = np.ones((max_time_step, patient_data.shape[1]), dtype=bool)
#             mask[original_length:, :] = False  # Mask the padded elements
#
#             data.append(torch.from_numpy(patient_data))
#             labels.append(sepsis)
#             masks.append(torch.from_numpy(mask))
#
#         logging.info(f"Total number of samples after applying window method: ({len(data)})")
#         logging.info(f"Distribution of Sepsis:\n{pd.Series(labels).value_counts()}")
#
#         return data, labels, masks
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, item):
#         return self.data[item], self.labels[item], self.mask[item]

import torch
import numpy as np
import logging
import tqdm
from torch.utils.data import Dataset


class DatasetWithPaddingMasking(Dataset):
    def __init__(self, training_examples_list, lengths_list, is_sepsis):
        self.data, self.labels, self.mask = self._create_dataset(training_examples_list, lengths_list, is_sepsis)

    def _create_dataset(self, training_examples_list, lengths_list, is_sepsis):
        logging.info(f"Input features ({len(training_examples_list[0].columns)}): {training_examples_list[0].columns}")
        data, labels, masks = [], [], []

        max_time_step = 336  # The maximum sequence length for padding

        for patient_data, length, sepsis in tqdm.tqdm(zip(training_examples_list, lengths_list, is_sepsis),
                                                      desc="Padding...", total=len(training_examples_list)):

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
    assert len(is_sepsis_file) == 20336, f"is_sepsis.txt didn't load properly"

    train_temp, test = train_test_split(is_sepsis_file, test_size=0.2, random_state=42)
    train, val = train_test_split(train_temp, test_size=0.2, random_state=42)

    train_indices = train.index.values
    val_indices = val.index.values
    test_indices = test.index.values

    return train_indices, val_indices, test_indices


class DatasetWithPaddingAndLengths(Dataset):
    def __init__(self, training_examples_list, lengths_list, is_sepsis):
        self.data, self.labels, self.lengths = self._create_dataset(training_examples_list, lengths_list, is_sepsis)

    def _create_dataset(self, training_examples_list, lengths_list, is_sepsis):
        logging.info(f"Input features ({len(training_examples_list[0].columns)}): {training_examples_list[0].columns}")
        data, labels, lengths = [], [], []
        max_time_step = 336
        for patient_data, sepsis, length in tqdm.tqdm(zip(training_examples_list, is_sepsis, lengths_list),
                                                      desc="Padding...",
                                                      total=len(training_examples_list)):
            patient_data = patient_data.drop(['PatientID', 'SepsisLabel'], axis=1)
            pad = (max_time_step - len(patient_data), 0)
            patient_data = np.pad(patient_data, pad_width=((0, pad[0]), (0, 0)), mode='constant').astype(np.float32)

            data.append(torch.from_numpy(patient_data))
            labels.append(sepsis)
            lengths.append(length)

        logging.info(f"Total number of samples after applying window method: ({len(data)})")
        logging.info(f"Distribution of Sepsis:\n{pd.Series(labels).value_counts()}")

        return data, labels, lengths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item], self.lengths[item]


# class DatasetWithPaddingAndLengths(Dataset):
#     """
#     Labels are saved as (timesteps,) instead of single value
#     """
#     def __init__(self, training_examples_list, lengths_list, is_sepsis):
#         self.data, self.labels, self.lengths = self._create_dataset(training_examples_list, lengths_list, is_sepsis)
#
#     def _create_dataset(self, training_examples_list, lengths_list, is_sepsis):
#         logging.info(f"Input features ({len(training_examples_list[0].columns)}): {training_examples_list[0].columns}")
#         data, labels, lengths = [], [], []
#         max_time_step = 336
#         for patient_data, sepsis, length in tqdm.tqdm(zip(training_examples_list, is_sepsis, lengths_list),
#                                                       desc="Padding...",
#                                                       total=len(training_examples_list)):
#             y_values = torch.from_numpy(patient_data['SepsisLabel'].values)
#             patient_data = patient_data.drop(['PatientID', 'SepsisLabel'], axis=1)
#             pad = (max_time_step - len(patient_data), 0)
#             patient_data = np.pad(patient_data, pad_width=((0, pad[0]), (0, 0)), mode='constant').astype(np.float32)
#
#             data.append(torch.from_numpy(patient_data))
#             labels.append(y_values)
#             lengths.append(length)
#
#         # logging.info(f"Total number of samples after applying window method: ({len(data)})")
#         # logging.info(f"Distribution of Sepsis:\n{pd.Series(labels).value_counts()}")
#
#         return data, labels, lengths
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, item):
#         return self.data[item], self.labels[item], self.lengths[item]


def save_train_means(train_samples):
    # mean_imputation = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'FiO2', 'pH', 'PaCO2', 'SaO2',
    #                    'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Creatinine', 'Bilirubin_direct', 'Glucose',
    #                    'Lactate', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'Fibrinogen', 'Platelets', 'Age']
    mean_imputation = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'FiO2', 'pH', 'PaCO2', 'SaO2',
                       'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Creatinine', 'Glucose',
                       'Lactate', 'Bilirubin_total', 'Hct', 'Hgb', 'Platelets', 'Age']
    median_imputation = ['HCO3', 'Chloride', 'Magnesium', 'Phosphate', 'Potassium', 'PTT', 'WBC', 'HospAdmTime',
                         'ICULOS']
    category_imputation = ['Gender', 'Unit1', 'Unit2']

    features_list = get_features(case=2) # Case 2 removes ['Bilirubin_direct', 'TroponinI', 'Fibrinogen'] features

    features = []
    for each_list in features_list:
        features.extend(each_list)

    combined_dataset = pd.concat(train_samples)
    features.extend(['SepsisLabel', 'PatientID'])
    combined_dataset = combined_dataset[features]

    means = combined_dataset[mean_imputation].mean(axis=0, skipna=True).round(2)
    medians = combined_dataset[median_imputation].median(axis=0, skipna=True).round(2)

    # For preprocessing (during evaluation)
    # combined_dataset['Unit1'] = combined_dataset['Unit1'].fillna(0)
    # combined_dataset['Unit2'] = combined_dataset['Unit2'].fillna(0)
    #
    # combined_dataset.loc[combined_dataset['Unit1'] == 0, 'Unit2'] = 1
    # combined_dataset.loc[combined_dataset['Unit2'] == 0, 'Unit1'] = 1
    #
    # combined_dataset['Gender'] = combined_dataset['Gender'].fillna(0)

    # save means
    means.to_csv('means.csv', index=mean_imputation)
    medians.to_csv('medians.csv', index=median_imputation)

    print(f"Saving means and medians of train set...! (for evaluation)")
    logging.info(f"Saving means and medians of train set...! (for evaluation)")


def make_loader(examples, lengths_list, is_sepsis, batch_size, mode, num_workers=4, train_indicies=None,
                test_indicies=None, val_indicies=None, select_important_features=False, include_val=False):
    # Loading data from given indicies
    if train_indicies is None and test_indicies is None:
        print("Loading data from pre-defined indicies")
        # Checking for validation set
        if include_val:
            print(f"Creating train, val and test sets.")
            logging.info(f"Creating train, val and test sets.")
            train_indicies, val_indicies, test_indicies = get_train_val_test_indices()
        else:
            print(f"Creating train and test sets.")
            logging.info(f"Creating train and test sets.")
            train_indicies, test_indicies = get_train_test_indicies()

    # Loading default data and using entire indicies
    else:
        print("Loading train and test from given indicies")
        if include_val:
            print(f"Creating train, val and test sets.")
            logging.info(f"Creating train, val and test sets.")

    # If validation set is required, load train, val, and test
    if include_val:
        train_samples = [examples[idx] for idx in train_indicies]
        val_samples = [examples[idx] for idx in val_indicies]
        test_samples = [examples[idx] for idx in test_indicies]

        train_lengths_list = [lengths_list[idx] for idx in train_indicies]
        val_lengths_list = [lengths_list[idx] for idx in val_indicies]
        test_lengths_list = [lengths_list[idx] for idx in test_indicies]

        is_sepsis_train = [is_sepsis[idx] for idx in train_indicies]
        is_sepsis_val = [is_sepsis[idx] for idx in val_indicies]
        is_sepsis_test = [is_sepsis[idx] for idx in test_indicies]

    # When validation set is not required, load train and test
    else:
        train_samples = [examples[idx] for idx in train_indicies]
        test_samples = [examples[idx] for idx in test_indicies]

        train_lengths_list = [lengths_list[idx] for idx in train_indicies]
        test_lengths_list = [lengths_list[idx] for idx in test_indicies]

        is_sepsis_train = [is_sepsis[idx] for idx in train_indicies]
        is_sepsis_test = [is_sepsis[idx] for idx in test_indicies]

    # Select features that are important
    if select_important_features:
        print(f"Selecting important features: Original no.: {train_samples[0].shape[1]}")

        train_samples = [patient_data[lgbm_features + ['PatientID', 'SepsisLabel']] for patient_data in train_samples]
        # If validation set is required, filter features.
        if include_val:
            val_samples = [patient_data[lgbm_features + ['PatientID', 'SepsisLabel']] for patient_data in val_samples]
        test_samples = [patient_data[lgbm_features + ['PatientID', 'SepsisLabel']] for patient_data in test_samples]

        print(f"Selected important features: Now: {train_samples[0].shape[1]}")

    # Saving means of train samples. (for evaluation)
    save_train_means(train_samples)

    # Operations on dataset
    if mode == "window":
        train_dataset = DatasetWithWindows(training_examples_list=train_samples, lengths_list=train_lengths_list,
                                           is_sepsis=is_sepsis_train, window_size=8, step_size=6)
        test_dataset = DatasetWithWindows(training_examples_list=test_samples, lengths_list=test_lengths_list,
                                          is_sepsis=is_sepsis_test, window_size=8, step_size=6)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        if include_val:
            val_dataset = DatasetWithWindows(training_examples_list=val_samples, lengths_list=val_lengths_list,
                                             is_sepsis=is_sepsis_val, window_size=8, step_size=6)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            logging.info(f"Window size: {train_dataset.window_size} & Step size: {train_dataset.step_size}")

            logging.info(f"Num of training examples: {len(train_dataset)}")
            logging.info(f"Num of validation examples: {len(val_dataset)}")
            logging.info(f"Num of test examples: {len(test_dataset)}")

            return train_loader, val_loader, test_loader, train_indicies, val_indicies, test_indicies

        logging.info(f"Window size: {train_dataset.window_size} & Step size: {train_dataset.step_size}")

    elif mode == "padding":
        train_dataset = DatasetWithPadding(training_examples_list=train_samples, lengths_list=train_lengths_list,
                                           is_sepsis=is_sepsis_train)
        test_dataset = DatasetWithPadding(training_examples_list=test_samples, lengths_list=test_lengths_list,
                                          is_sepsis=is_sepsis_test, )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        if include_val:
            val_dataset = DatasetWithPadding(training_examples_list=val_samples, lengths_list=val_lengths_list,
                                             is_sepsis=is_sepsis_val, )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            logging.info(f"Num of training examples: {len(train_dataset)}")
            logging.info(f"Num of validation examples: {len(val_dataset)}")
            logging.info(f"Num of test examples: {len(test_dataset)}")

            return train_loader, val_loader, test_loader, train_indicies, val_indicies, test_indicies

    elif mode == "default":
        train_dataset = DefaultDataset(training_examples_list=train_samples, lengths_list=train_lengths_list,
                                       is_sepsis=is_sepsis_train)
        test_dataset = DefaultDataset(training_examples_list=test_samples, lengths_list=test_lengths_list,
                                      is_sepsis=is_sepsis_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 collate_fn=collate_fn)

        if include_val:
            val_dataset = DatasetWithPadding(training_examples_list=val_samples, lengths_list=val_lengths_list,
                                             is_sepsis=is_sepsis_val, window_size=8, step_size=6)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            logging.info(f"Window size: {train_dataset.window_size} & Step size: {train_dataset.step_size}")

            logging.info(f"Num of training examples: {len(train_dataset)}")
            logging.info(f"Num of validation examples: {len(val_dataset)}")
            logging.info(f"Num of test examples: {len(test_dataset)}")

            return train_loader, val_loader, test_loader, train_indicies, val_indicies, test_indicies

    elif mode == "padding_masking":
        train_dataset = DatasetWithPaddingMasking(training_examples_list=train_samples, lengths_list=train_lengths_list,
                                                  is_sepsis=is_sepsis_train)
        test_dataset = DatasetWithPaddingMasking(training_examples_list=test_samples, lengths_list=test_lengths_list,
                                                 is_sepsis=is_sepsis_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        if include_val:
            val_dataset = DatasetWithPaddingMasking(training_examples_list=val_samples, lengths_list=val_lengths_list,
                                                    is_sepsis=is_sepsis_val, )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            logging.info(f"Num of training examples: {len(train_dataset)}")
            logging.info(f"Num of test examples: {len(test_dataset)}")

            return train_loader, val_loader, test_loader, train_indicies, val_indicies, test_indicies

    elif mode == "padding_and_lengths":
        train_dataset = DatasetWithPaddingAndLengths(training_examples_list=train_samples,
                                                     lengths_list=train_lengths_list,
                                                     is_sepsis=is_sepsis_train)
        test_dataset = DatasetWithPaddingAndLengths(training_examples_list=test_samples, lengths_list=test_lengths_list,
                                                    is_sepsis=is_sepsis_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        logging.info(f"Batch size is {batch_size}")

        if include_val:
            val_dataset = DatasetWithPaddingAndLengths(training_examples_list=val_samples,
                                                       lengths_list=val_lengths_list,
                                                       is_sepsis=is_sepsis_val, )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            # logging.info(f"Window size: {train_dataset.window_size} & Step size: {train_dataset.step_size}")

            logging.info(f"Num of training examples: {len(train_dataset)}")
            logging.info(f"Num of test examples: {len(test_dataset)}")

            return train_loader, val_loader, test_loader, train_indicies, val_indicies, test_indicies

    logging.info(f"Num of training examples: {len(train_dataset)}")
    logging.info(f"Num of test examples: {len(test_dataset)}")

    return train_loader, test_loader, train_indicies, test_indicies


def initialize_experiment(data_file):
    data_file = "final_dataset.pickle"

    print(f"Dataset used: {data_file}")

    # [[patient1], [patient2], [patient3], ..., [patientN]]
    training_examples = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', data_file))

    with open(os.path.join(project_root(), 'data', 'processed', 'lengths.txt')) as f:
        lengths_list = [int(length) for length in f.read().splitlines()]

    with open(os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt')) as f:
        is_sepsis = [int(is_sep) for is_sep in f.read().splitlines()]

    return training_examples, lengths_list, is_sepsis

# if __name__ == '__main__':
#     training_examples, lengths_list, is_sepsis = initialize_experiment()
#     train_loader, test_loader = make_loader(training_examples, lengths_list, is_sepsis, batch_size=128)
#
#     for idx, patient_data in enumerate(train_loader):
#         if idx == 1:
#             print(patient_data)
#             break
