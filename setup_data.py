import os
import pickle

import numpy as np
import pandas as pd
import tqdm

from sklearn.preprocessing import StandardScaler

from utils.path_utils import project_root

from utils.add_features import *


class DataSetup:
    def __init__(self):

        self.data_paths = [os.path.join(project_root(), 'physionet.org', 'files', 'challenge-2019', '1.0.0', 'training', 'training_setA'),
                        #    os.path.join(project_root(), 'physionet.org', 'files', 'challenge-2019', '1.0.0', 'training', 'training_setB')
                           ]

        self.destination_path = os.path.join(project_root(), 'data', 'csv')
        self.hdf_path = os.path.join(project_root(), 'data', 'hdf')
        self.csv_path = os.path.join(project_root(), 'data', 'csv')

    def convert_to_csv(self):
        for data_path in self.data_paths:
            training_files = [file for file in os.listdir(data_path) if file.endswith('.psv')]
            training_files.sort()

            for i, file in enumerate(
                    tqdm.tqdm(training_files, desc="Converting psv to csv", total=len(training_files))):
                try:
                    temp = pd.read_csv(os.path.join(data_path, file), sep='|')
                    temp['PatientID'] = i
                    temp.to_csv(os.path.join(self.destination_path, file.replace('.psv', '.csv')), sep=',', index=False)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

        print("Conversion complete.")

    def rewrite_csv(self, training_files):

        lengths, is_sepsis, all_data, ind = [], [], np.zeros((1552210, 42)), 0  # (num_rows, features)
        training_examples = []
        for i, training_file in enumerate(
                tqdm.tqdm(training_files, desc="Creating train.pickle, lengths_list, is_sepsis files",
                          total=len(training_files))):
            example = pd.read_csv(training_file, sep=',')
            example['PatientID'] = i

            training_examples.append(example)
            is_sepsis.append(1 if 1 in example['SepsisLabel'].values else 0)
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
            for training_file in tqdm.tqdm(training_files, desc="Mean Imputation", total=len(training_files)):
                example = pd.read_csv(training_file, sep=',')
                example.fillna(means, inplace=True)
                training_examples.append(example)

                example.to_csv(os.path.join(self.destination_path, training_file), index=False)
            with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
                pickle.dump(training_examples, f)

        elif method == 'median':
            dataset_name = 'training_median.pickle'
            print(f"Filling missing values with median")
            training_examples = []
            medians = all_data.median(axis=0, skipna=True)
            for training_file in tqdm.tqdm(training_files, desc="Median Imputation", total=len(training_files)):
                example = pd.read_csv(training_file, sep=',')
                example.fillna(medians, inplace=True)
                training_examples.append(example)

                example.to_csv(os.path.join(self.destination_path, training_file), index=False)
            with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
                pickle.dump(training_examples, f)

        elif method == 'zeros':
            dataset_name = 'training_zeros.pickle'
            print(f"Filling missing values with zeros")
            training_examples = []
            for training_file in tqdm.tqdm(training_files, desc="Zero Imputation", total=len(training_files)):
                example = pd.read_csv(training_file, sep=',')
                # example = example.drop(['SepsisLabel'], axis=1)
                example.fillna(0, inplace=True)
                training_examples.append(example)

                example.to_csv(os.path.join(self.destination_path, training_file), index=False)
            with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
                pickle.dump(training_examples, f)

        else:
            dataset_name = 'training_ffill_bfill_zeros.pickle'
            print(f"Filling missing values with ffill, bfill, and zeros")
            training_examples = []
            for training_file in tqdm.tqdm(training_files, desc="Ffill, Bfill, Zeros Imputation",
                                           total=len(training_files)):
                example = pd.read_csv(training_file, sep=',')
                example.ffill(inplace=True)
                example.bfill(inplace=True)
                example.fillna(value=0, inplace=True)
                training_examples.append(example)

                example.to_csv(os.path.join(self.destination_path, training_file), index=False)
            with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
                pickle.dump(training_examples, f)

        return dataset_name

    # def add_lag_features(self, training_examples):
    #     # TODO: Pending
    #     training_examples_lag = []
    #     for training_example in tqdm.tqdm(training_examples):
    #         lag_columns = [column + '_lag' for column in training_example.columns.values[:35]]
    #         lag_features = training_example.values[:-6, :35] - training_example.values[6: 35]
    #         training_example = pd.concat([training_example, pd.DataFrame(columns=lag_columns)])
    #
    #         training_examples_lag.append(training_example)
    #
    #     return training_examples_lag

    def add_additional_features(self, data):
        dataset_name = "final_dataset.pickle"
        training_examples = []
        for patient_id, patient_data in tqdm.tqdm(enumerate(data), desc="Adding additional features", total=len(data)):
            patient_data['MAP_SOFA'] = patient_data['MAP'].apply(map_sofa)
            patient_data['Bilirubin_total_SOFA'] = patient_data['Bilirubin_total'].apply(total_bilirubin_sofa)
            patient_data['Platelets_SOFA'] = patient_data['Platelets'].apply(platelets_sofa)
            patient_data['SOFA_score'] = patient_data.apply(sofa_score, axis=1)
            patient_data = detect_sofa_change(patient_data)

            patient_data['ResP_qSOFA'] = patient_data['Resp'].apply(respiratory_rate_qsofa)
            patient_data['SBP_qSOFA'] = patient_data['SBP'].apply(sbp_qsofa)
            patient_data['qSOFA_score'] = patient_data.apply(qsofa_score, axis=1)
            patient_data = detect_qsofa_change(patient_data)

            patient_data['qSOFA_indicator'] = patient_data.apply(q_sofa_indicator, axis=1)  # Sepsis detected
            patient_data['SOFA_indicator'] = patient_data.apply(sofa_indicator, axis=1)  # Organ Dysfunction occurred
            patient_data['Mortality_sofa'] = patient_data.apply(mortality_sofa, axis=1)  # Morality rate

            patient_data['Temp_sirs'] = patient_data['Temp'].apply(temp_sirs)
            patient_data['HR_sirs'] = patient_data['HR'].apply(heart_rate_sirs)
            patient_data['Resp_sirs'] = patient_data['Resp'].apply(resp_sirs)
            patient_data['paco2_sirs'] = patient_data['PaCO2'].apply(resp_sirs)
            patient_data['wbc_sirs'] = patient_data['WBC'].apply(wbc_sirs)

            patient_data = t_suspicion(patient_data)
            patient_data = t_sofa(patient_data)
            patient_data['t_sepsis'] = patient_data.apply(t_sepsis, axis=1)

            training_examples.append(patient_data)

            patient_data.to_csv(os.path.join(project_root(), 'data', 'test', f"patient_id_{patient_id}.csv"), index=False)

        with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
            pickle.dump(training_examples, f)


if __name__ == '__main__':

    setup = DataSetup()

    # Converts psv to csv
    setup.convert_to_csv()

    # Rewriting data
    csv_path = os.path.join(project_root(), 'data', 'csv')
    training_files = [os.path.join(csv_path, f) for f in os.listdir(csv_path) if f.endswith('.csv')]
    training_files.sort()
    setup.rewrite_csv(training_files=training_files)

    # Standardising the data and Filling missing values and save csv files back
    data_file_name = setup.fill_missing_values(method='None', training_files=training_files)

    # Add features
    dataset = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', 'training_ffill_bfill_zeros.pickle'))
    setup.add_additional_features(data=dataset)

    # Remove unwanted features


#     # Adding lag features
#     # training_examples = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', data_file_name))
#     # setup.add_lag_features(training_examples)
#     # with open(os.path.join(project_root(), 'data', 'processed', data_file_name+'_lag'), 'wb') as f:
#     #     pickle.dump(training_examples, f)
