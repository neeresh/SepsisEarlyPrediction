import os
import pickle

import numpy as np
import pandas as pd
import tqdm

from sklearn.preprocessing import StandardScaler

from utils.path_utils import project_root


class DataSetup:
    def __init__(self):

        self.data_paths = [os.path.join(project_root(), 'physionet.org', 'files', 'challenge-2019', '1.0.0', 'training',
                                        'training_setA'),
                           os.path.join(project_root(), 'physionet.org', 'files', 'challenge-2019', '1.0.0', 'training',
                                        'training_setB')]

        self.destination_path = os.path.join(project_root(), 'data', 'csv')
        self.hdf_path = os.path.join(project_root(), 'data', 'hdf')
        self.csv_path = os.path.join(project_root(), 'data', 'csv')

    def convert_to_csv(self):
        for data_path in self.data_paths:
            training_files = [file for file in os.listdir(data_path) if file.endswith('.psv')]
            training_files.sort()

            for file in tqdm.tqdm(training_files):
                try:
                    temp = pd.read_csv(os.path.join(data_path, file), sep='|')
                    temp.to_csv(os.path.join(self.destination_path, file.replace('.psv', '.csv')), sep=',', index=False)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

        print("Conversion complete.")

    def rewrite_csv(self, training_files):

        lengths, is_sepsis, all_data, ind = [], [], np.zeros((1552210, 42)), 0  # (num_rows, features)
        training_examples = []
        for i, training_file in enumerate(tqdm.tqdm(training_files)):
            example = pd.read_csv(training_file, sep=',')
            example['seg_id'] = i

            training_examples.append(example)
            is_sepsis.append(1 if 1 in example['SepsisLabel'].values else 0)  # If patient diagnosed with sepsis or not
            lengths.append(len(example))

            all_data[ind: ind + len(example), :] = example.values
            ind += len(example)

        all_data = pd.DataFrame(all_data, columns=example.columns.values, index=None)
        all_data.to_hdf(os.path.join(project_root(), 'data', 'processed', 'training_concatenated.hdf'), key='df')
        all_data.to_csv(os.path.join(project_root(), 'data', 'processed', 'training_concatenated.csv'), index=False)

        with open(os.path.join(project_root(), 'data', 'processed', 'lengths.txt'), 'w') as f:
            [f.write(f'{l}\n') for l in lengths]

        with open(os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt'), 'w') as f:
            [f.write(f'{l}\n') for l in is_sepsis]

        with open(os.path.join(project_root(), 'data', 'processed', 'training_raw.pickle'), 'wb') as f:
            pickle.dump(training_examples, f)

    def fill_missing_values(self, training_files, method):

        scalar = StandardScaler()

        all_data_path = os.path.join(os.path.join(project_root(), 'data', 'processed', 'training_concatenated.csv'))
        all_data = pd.read_csv(all_data_path)

        if method == 'mean':
            dataset_name = 'training_mean.pickle'
            print(f"Filling missing values with mean")
            training_examples = []
            means = all_data.mean(axis=0, skipna=True)
            for training_file in tqdm.tqdm(training_files):
                example = pd.read_csv(training_file, sep=',')
                example.fillna(means, inplace=True)
                training_examples.append(example)

            with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
                pickle.dump(training_examples, f)

        elif method == 'median':
            dataset_name = 'training_median.pickle'
            print(f"Filling missing values with median")
            training_examples = []
            medians = all_data.median(axis=0, skipna=True)
            for training_file in tqdm.tqdm(training_files):
                example = pd.read_csv(training_file, sep=',')
                example.fillna(medians, inplace=True)
                training_examples.append(example)

            with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
                pickle.dump(training_examples, f)

        elif method == 'zeros':
            dataset_name = 'training_zeros.pickle'
            print(f"Filling missing values with zeros")
            training_examples = []
            for training_file in tqdm.tqdm(training_files):
                example = pd.read_csv(training_file, sep=',')
                example.fillna(0, inplace=True)
                training_examples.append(example)

            with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
                pickle.dump(training_examples, f)

        else:
            dataset_name = 'training_ffill_bfill_zeros.pickle'
            print(f"Filling missing values with ffill, bfill, and zeros")
            training_examples = []
            for training_file in tqdm.tqdm(training_files):
                example = pd.read_csv(training_file, sep=',')
                example.ffill(inplace=True)
                example.bfill(inplace=True)
                example.fillna(value=0, inplace=True)
                training_examples.append(example)

            with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
                pickle.dump(training_examples, f)

        return dataset_name

    def add_lag_features(self, training_examples):
        # TODO: Pending
        training_examples_lag = []
        for training_example in tqdm.tqdm(training_examples):
            lag_columns = [column + '_lag' for column in training_example.columns.values[:35]]
            lag_features = training_example.values[:-6, :35] - training_example.values[6: 35]
            training_example = pd.concat([training_example, pd.DataFrame(columns=lag_columns)])

            training_examples_lag.append(training_example)

        return training_examples_lag


if __name__ == '__main__':
    setup = DataSetup()

    # Converts psv to csv
    # setup.convert_to_csv()

    # Rewriting data
    csv_path = os.path.join(project_root(), 'data', 'csv')
    training_files = [os.path.join(csv_path, f) for f in os.listdir(csv_path) if f.endswith('.csv')]
    training_files.sort()
    # setup.rewrite_csv(training_files=training_files)

    # Standardising the data and Filling missing values
    data_file_name = setup.fill_missing_values(method='None', training_files=training_files)

    # Adding lag features
    # training_examples = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', data_file_name))
    # setup.add_lag_features(training_examples)
    # with open(os.path.join(project_root(), 'data', 'processed', data_file_name+'_lag'), 'wb') as f:
    #     pickle.dump(training_examples, f)
