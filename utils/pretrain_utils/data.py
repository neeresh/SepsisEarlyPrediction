import numpy as np
import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import tqdm

from utils.path_utils import project_root

from torch.utils.data import Dataset


class Load_Dataset(Dataset):

    def __init__(self, dataset, TSlength_aligned, training_mode):

        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        # shuffle
        data = list(zip(X_train, y_train))
        np.random.shuffle(data)

        X_train, y_train = zip(*data)
        X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0).squeeze()

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        # if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
        #     X_train = X_train.permute(0, 2, 1)

        """Align the TS length between source and target datasets"""
        # X_train = X_train[:, :1, :int(config.TSlength_aligned)] # take the first 178 samples
        X_train = X_train[:, :, :int(TSlength_aligned)]

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def csv_to_pt(patient_files, lengths, is_sepsis, desc):
    all_patients = {'samples': [], 'labels': []}

    max_time_step = 336
    # print(len(patient_files), len(lengths), len(is_sepsis))
    for idx, (file, length, sepsis) in tqdm.tqdm(enumerate(zip(patient_files, lengths, is_sepsis)),
                                                 desc=f"{desc}",
                                                 total=len(patient_files)):

        pad_width = ((0, max_time_step - len(file)), (0, 0))
        file = np.pad(file, pad_width=pad_width, mode='constant').astype(np.float32)

        if len(file) == max_time_step:
            all_patients['samples'].append(torch.from_numpy(file).unsqueeze(0))
            all_patients['labels'].append(torch.tensor(sepsis, dtype=torch.float32).unsqueeze(0))
        else:
            raise ValueError(f"Length {length} does not match length of patient {idx} with length {len(file)}")

    # print('samples: ', type(all_patients['samples']), 'labels: ', type(all_patients['labels']))

    all_patients['samples'] = torch.cat(all_patients['samples'], dim=0)
    all_patients['labels'] = torch.cat(all_patients['labels'], dim=0)

    return {'samples': all_patients['samples'], 'labels': all_patients['labels']}, lengths, is_sepsis


def get_train_val_test_indices(sepsis_file, dset, test_perc=0.2, save_distributions=True):
    sepsis = pd.read_csv(os.path.join(project_root(), 'data', 'tl_datasets', f'{sepsis_file}'), header=None)

    train_indices, val_indices = train_test_split(sepsis, test_size=test_perc, random_state=2024)

    train_indices = train_indices.index.values
    val_indices = val_indices.index.values

    if save_distributions:
        train_dist = sepsis.iloc[train_indices].value_counts()
        val_dist = sepsis.iloc[val_indices].value_counts()

        train_dist_percentage = np.round(train_dist / len(sepsis.iloc[train_indices]), 2)
        val_dist_percentage = np.round(val_dist / len(sepsis.iloc[val_indices]), 2)

        pd.DataFrame(
            {
                'Train Images': train_dist, 'Validation Images': val_dist,
                'Train Distribution Percentage': train_dist_percentage,
                'Validation Distribution Percentage': val_dist_percentage,
            }
        ).to_csv(os.path.join(project_root(), 'results', f'distributions{dset}.csv'), index=False)

        # pd.read_csv(os.path.join(project_root(), 'results', 'distributions.csv'))

    return train_indices, val_indices


def get_pretrain_finetune_datasets():
    # Pre-training Indices
    pt_train_indices, pt_val_indices = get_train_val_test_indices(
        sepsis_file='is_sepsis_pretrain_A.txt', save_distributions=True,
        dset='Aa', test_perc=0.2)

    # Gathering files, lengths, and sepsis label
    pt_files = pd.read_pickle(os.path.join(project_root(), 'data', 'tl_datasets', 'final_dataset_pretrain_A.pickle'))
    pt_lengths = pd.read_csv(os.path.join(project_root(), 'data', 'tl_datasets', 'lengths_pretrain_A.txt'),
                             header=None)
    pt_sepsis = pd.read_csv(os.path.join(project_root(), 'data', 'tl_datasets', 'is_sepsis_pretrain_A.txt'),
                            header=None)

    # Checking whether the files are in same order or not
    pretrain_files = []
    for pdata, length in tqdm.tqdm(zip(pt_files, pt_lengths.values), desc="Checking Pre-training & Validation Files",
                                   total=len(pt_files)):
        plength = len(pdata)
        assert plength == length[0], f"{plength} doesn't match {length}"
        pretrain_files.append(pdata.drop(['PatientID', 'SepsisLabel'], axis=1))

    # Getting train and val
    pt_train = [pretrain_files[i] for i in pt_train_indices]
    pt_val = [pretrain_files[i] for i in pt_val_indices]

    pt_train_lengths = pt_lengths.iloc[pt_train_indices].values
    pt_val_lengths = pt_lengths.iloc[pt_val_indices].values

    pt_train_sepsis = pt_sepsis.iloc[pt_train_indices].values
    pt_val_sepsis = pt_sepsis.iloc[pt_val_indices].values

    pt_train, pt_train_lengths, pt_train_sepsis = csv_to_pt(pt_train, pt_train_lengths, pt_train_sepsis,
                                                            desc='PT Train Set')
    pt_val, pt_val_lengths, pt_val_sepsis = csv_to_pt(pt_val, pt_val_lengths, pt_val_sepsis, desc='PT Validation Set')

    # Fine-tuning
    test_indices, finetune_indices = get_train_val_test_indices(
        sepsis_file='is_sepsis_finetune_B.txt', save_distributions=True,
        dset='Bb')

    # Gathering files, lengths, and sepsis label
    test_setB = pd.read_pickle(os.path.join(project_root(), 'data', 'tl_datasets', 'final_dataset_finetune_B.pickle'))
    test_setB_lengths = pd.read_csv(os.path.join(project_root(), 'data', 'tl_datasets', 'lengths_finetune_B.txt'),
                                    header=None)
    test_setB_sepsis = pd.read_csv(os.path.join(project_root(), 'data', 'tl_datasets', 'is_sepsis_finetune_B.txt'),
                                   header=None)

    # Checking whether the files are in same order or not
    test_files = []
    for pdata, length in tqdm.tqdm(zip(test_setB, test_setB_lengths.values), desc="Checking Fine-tuning & Test Files",
                                   total=len(test_setB)):
        plength = len(pdata)
        assert plength == length[0], f"{plength} doesn't match {length}"
        test_files.append(pdata.drop(['PatientID', 'SepsisLabel'], axis=1))

    # Getting finetune and test sets
    finetune = [test_files[i] for i in finetune_indices]
    test = [test_files[i] for i in test_indices]

    finetune_lengths = test_setB_lengths.iloc[finetune_indices].values
    test_lengths = test_setB_lengths.iloc[test_indices].values

    finetune_sepsis = test_setB_sepsis.iloc[finetune_indices].values
    test_sepsis = test_setB_sepsis.iloc[test_indices].values

    finetune, finetune_lengths, finetune_sepsis = csv_to_pt(finetune, finetune_lengths, finetune_sepsis,
                                                            desc="Fine-tuning Set")
    test, test_lengths, test_sepsis = csv_to_pt(test, test_lengths, test_sepsis, desc="Test Set")

    print("Pre-training samples: ", pt_train['samples'].shape, "Validation samples: ", pt_val['samples'].shape)
    print("Fine-tuning samples: ", finetune['samples'].shape, "Test samples: ", test['samples'].shape)

    return pt_train, pt_val, finetune, test


# pt_train, pt_val, finetune, test = get_pretrain_finetune_datasets()

def get_pretrain_finetune_test_datasets():

    # # Pre-training Indices
    # pt_train_indices, pt_val_indices = get_train_val_test_indices(
    #     sepsis_file='is_sepsis_pretrain_A.txt', save_distributions=True,
    #     dset='Aa', test_perc=0.01)

    # Gathering files, lengths, and sepsis label
    pt_files = pd.read_pickle(os.path.join(project_root(), 'data', 'tl_datasets', 'final_dataset_pretrain_A.pickle'))
    pt_lengths = pd.read_csv(os.path.join(project_root(), 'data', 'tl_datasets', 'lengths_pretrain_A.txt'),
                             header=None)
    pt_sepsis = pd.read_csv(os.path.join(project_root(), 'data', 'tl_datasets', 'is_sepsis_pretrain_A.txt'),
                            header=None)

    # Checking whether the files are in same order or not
    pretrain_files = []
    for pdata, length in tqdm.tqdm(zip(pt_files, pt_lengths.values), desc="Checking Pre-training",
                                   total=len(pt_files)):
        plength = len(pdata)
        assert plength == length[0], f"{plength} doesn't match {length}"
        pretrain_files.append(pdata.drop(['PatientID', 'SepsisLabel'], axis=1))

    # Getting train and val
    pt_train = pretrain_files
    pt_train_lengths = pt_lengths.values
    pt_train_sepsis = pt_sepsis.values

    pt_train, pt_train_lengths, pt_train_sepsis = csv_to_pt(pt_train, pt_train_lengths, pt_train_sepsis,
                                                            desc='PT Train Set')
    # pt_val, pt_val_lengths, pt_val_sepsis = csv_to_pt(pt_val, pt_val_lengths, pt_val_sepsis, desc='PT Validation Set')

    # Fine-tuning
    test_indices, finetune_indices = get_train_val_test_indices(
        sepsis_file='is_sepsis_finetune_B.txt', save_distributions=True,
        dset='Bb')

    # Gathering files, lengths, and sepsis label
    test_setB = pd.read_pickle(os.path.join(project_root(), 'data', 'tl_datasets', 'final_dataset_finetune_B.pickle'))
    test_setB_lengths = pd.read_csv(os.path.join(project_root(), 'data', 'tl_datasets', 'lengths_finetune_B.txt'),
                                    header=None)
    test_setB_sepsis = pd.read_csv(os.path.join(project_root(), 'data', 'tl_datasets', 'is_sepsis_finetune_B.txt'),
                                   header=None)

    # Checking whether the files are in same order or not
    test_files = []
    for pdata, length in tqdm.tqdm(zip(test_setB, test_setB_lengths.values), desc="Checking Fine-tuning & Test Files",
                                   total=len(test_setB)):
        plength = len(pdata)
        assert plength == length[0], f"{plength} doesn't match {length}"
        test_files.append(pdata.drop(['PatientID', 'SepsisLabel'], axis=1))

    # Getting finetune and test sets
    finetune = [test_files[i] for i in finetune_indices]
    test = [test_files[i] for i in test_indices]

    finetune_lengths = test_setB_lengths.iloc[finetune_indices].values
    test_lengths = test_setB_lengths.iloc[test_indices].values

    finetune_sepsis = test_setB_sepsis.iloc[finetune_indices].values
    test_sepsis = test_setB_sepsis.iloc[test_indices].values

    finetune, finetune_lengths, finetune_sepsis = csv_to_pt(finetune, finetune_lengths, finetune_sepsis,
                                                            desc="Fine-tuning Set")
    test, test_lengths, test_sepsis = csv_to_pt(test, test_lengths, test_sepsis, desc="Test Set")

    print("Pre-training samples: ", pt_train['samples'].shape)
    print("Fine-tuning samples: ", finetune['samples'].shape, "Test samples: ", test['samples'].shape)

    # Saving original test psv files (for evaluation)
    test_setB_all_files = os.path.join(project_root(), 'physionet.org', 'files', 'challenge-2019', '1.0.0', 'training',
                                       'training_setB')
    test_setB_files = os.listdir(os.path.join(test_setB_all_files))
    test_setB_files.sort()
    test_setB_files.remove('index.html')

    test_setB_files = [test_setB_files[i] for i in test_indices]
    save_path = os.path.join(project_root(), 'data', 'test_data', 'simmtm', 'psv_files')
    for pidx in tqdm.tqdm(test_setB_files, desc="Saving test data (as psv files)", total=len(test_setB_files)):
        pdata = pd.read_csv(os.path.join(test_setB_all_files, pidx), sep='|')
        pdata.to_csv(os.path.join(save_path, pidx), sep='|', index=False)

    return pt_train, finetune, test

# pt_train, finetune, test = get_pretrain_finetune_datasets()
