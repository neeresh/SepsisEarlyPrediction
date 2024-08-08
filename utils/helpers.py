import random

import numpy as np
import pandas as pd

import glob
import os

from matplotlib import pyplot as plt
import seaborn as sns


def humansize(bytes):
    i = 0
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    while bytes >= 1024 and i < len(suffixes) - 1:
        bytes /= 1024.
        i += 1
    f = ('%.2f' % bytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])


def load_data(directories: list, target_label: str, x_train_y_train=False) -> pd.DataFrame:
    patient_id, df_list, record_lengths = 0, [], []
    for directory in directories:
        all_files = glob.glob(os.path.join(directory, '*.psv'))
        counter = 0
        for file in all_files:
            with open(file, 'r') as f:
                header = f.readline().strip()
                column_names = header.split('|')
                data = np.loadtxt(f, delimiter='|')
                data = pd.DataFrame(data, columns=column_names)
                data['PatientID'] = patient_id
                df_list.append(data)

            # Number of recording for each patient
            record_lengths.append(len(data))

            # Counting records
            counter += 1
            patient_id += 1

        print(f"Loaded {counter} records from {directory}.")

    df = pd.concat(df_list, ignore_index=True)
    print(f"Total number of records: {patient_id} (Check using: df['PatientID'].nunique())")
    print(f"Memory consumed: {humansize(df.memory_usage(index=True, deep=True).sum())}")

    if x_train_y_train:
        X_train = df.drop([target_label], axis=1)
        y_train = df[target_label]

        return X_train, y_train, record_lengths

    return df, record_lengths


def plot_target_classes(dataset, record_lengths):
    """
    Example:
        directories = ['./physionet.org/files/challenge-2019/1.0.0/training/training_setA',
                   './physionet.org/files/challenge-2019/1.0.0/training/training_setB']
        dataset, record_lengths = load_data(directories=directories, target_label='SepsisLabel')
        plot_target_classes(dataset, record_lengths)
    """
    start_length, ones, zeros = 0, 0, 0
    for idx, length in enumerate(record_lengths):
        end_length = start_length + length
        if dataset['SepsisLabel'][start_length:end_length].isin([1.0]).any():
            ones += 1
        else:
            zeros += 1
        start_length = end_length

    data = {'Category': ['Sepsis', 'Non-Sepsis'], 'Count': [ones, zeros]}
    plot_df = pd.DataFrame(data)
    ax = sns.barplot(x='Category', y='Count', data=plot_df)
    plt.title('Counts of 1.0s and 0.0s in SepsisLabel')
    total = ones + zeros
    for p in ax.patches:
        percentage = '{:.3f}%'.format(100 * p.get_height() / total)
        ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()), fontsize='x-small',
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.show()


def missing_percentage(dataset):
    """
    Example:
        directories = ['./physionet.org/files/challenge-2019/1.0.0/training/training_setA',
                   './physionet.org/files/challenge-2019/1.0.0/training/training_setB']
        dataset, record_lengths = load_data(directories=directories, target_label='SepsisLabel')
        missing_percentage(dataset)
    """
    missing_percent = dataset.isnull().mean() * 100
    missing_df = pd.DataFrame({
        'Features': missing_percent.index,
        'Percentage': missing_percent.values
    })

    plt.figure(figsize=(10, 4))
    sns.barplot(x='Features', y='Percentage', data=missing_df)
    plt.title('Percentage of Missing Values')
    plt.xticks(rotation=90)
    plt.axhline(90, color='r', linestyle='--')
    plt.axhline(95, color='r', linestyle='--')
    plt.axhline(99, color='r', linestyle='--')
    plt.text(42.5, 88, '90%', color='r', ha='center', fontdict={'size': 8})
    plt.text(42.5, 93, '95%', color='r', ha='center', fontdict={'size': 8})
    plt.text(42.5, 98, '99%', color='r', ha='center', fontdict={'size': 8})

    plt.show()


def get_features(case):
    """
    Case = 1, 2, 3, ...
    """
    vital_signs = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
    laboratory_values = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium',
                         'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
                         'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen',
                         'Platelets']
    demographics = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']

    if case == 1:
        vital_signs.remove('EtCO2')  # >90% NaN
        laboratory_values = []  # removing all (>90% NaNs)
        demographics = demographics  # None removed
        print(f"Total number of features: {len(vital_signs) + len(demographics)}")

    if case == 2:
        laboratory_values.remove('Bilirubin_direct')
        laboratory_values.remove('TroponinI')
        laboratory_values.remove('Fibrinogen')

    else:
        print(f"Total number of features: {len(vital_signs) + len(laboratory_values) + len(demographics)}")

    return vital_signs, laboratory_values, demographics


def plot_random_patient_recordings(dataset, feature='HR', num_plots=5, fill_method='ffill'):
    """
    Example:
        selected_patient_ids = plot_random_patient_recordings(dataset, feature='HR', num_plots=3, fill_method='ffill')
    """
    # selected_patient_ids = [14019, 24020, 39830]
    selected_patient_ids = random.sample(list(dataset['PatientID'].unique()), num_plots)
    fig, axes = plt.subplots(num_plots, 1, figsize=(15, 2 * num_plots), sharex=True)

    for i, patient_id in enumerate(selected_patient_ids):
        patient_data = dataset[dataset['PatientID'] == patient_id]
        axes[i].plot(patient_data['ICULOS'], patient_data[feature], label='Original Data', color='blue')

        # Imputing
        if fill_method.lower() == 'ffill':
            temp_data = patient_data.ffill()
        elif fill_method.lower() == 'bfill':
            temp_data = patient_data.bfill()

        axes[i].plot(temp_data['ICULOS'], temp_data[feature], label='Imputed Data', linestyle='--', color='orange')

        axes[i].set_title(f'Patient {patient_id}')
        axes[i].set_xlabel('ICULOS (hours)')
        axes[i].set_ylabel(feature)
        axes[i].legend()

    plt.tight_layout()
    plt.show()

    return selected_patient_ids
