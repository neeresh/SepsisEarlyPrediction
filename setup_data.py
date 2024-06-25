import os
import pickle

import numpy as np
import pandas as pd
import tqdm

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.path_utils import project_root

from utils.add_features import *
from utils.helpers import get_features


class DataSetup:
    def __init__(self):

        self.data_paths = [os.path.join(project_root(), 'physionet.org', 'files', 'challenge-2019', '1.0.0', 'training', 'training_setA'),
                           os.path.join(project_root(), 'physionet.org', 'files', 'challenge-2019', '1.0.0', 'training', 'training_setB')
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
            
            print(f"fill_missing_values() -> Dataset is saved under the name: {dataset_name}")

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
            
            print(f"fill_missing_values() -> Dataset is saved under the name: {dataset_name}")

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
            
            print(f"fill_missing_values() -> Dataset is saved under the name: {dataset_name}")

        elif method == 'rolling':
            dataset_name = 'training_rolling.pickle'
            print(f"Filling vital featuers using rolling method")
            print(f"Filling all the other values with the ffill and bfill method")

            # vital_signs = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
            vital_features, _, _ = get_features(case=1)

            means = all_data.mean(axis=0, skipna=True)
            training_examples = []
            for training_file in tqdm.tqdm(training_files, desc="Filling missing using rolling & bfill(), and means", total=len(training_files)):
                example = pd.read_csv(training_file, sep=',')
                for feature in vital_features:
                    example[feature] = example[feature].rolling(window=6, min_periods=1).mean().bfill()
                example.fillna(means, inplace=True)
                training_examples.append(example)

                example.to_csv(os.path.join(self.destination_path, training_file), index=False)

            with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
                pickle.dump(training_examples, f)
            
            print(f"fill_missing_values() -> Dataset is saved under the name: {dataset_name}")

        else:
            dataset_name = 'training_ffill_bfill_zeros.pickle'
            print(f"Filling missing values with ffill, bfill, and zeros")
            training_examples = []
            # means = all_data.mean(axis=0, skipna=True)
            for training_file in tqdm.tqdm(training_files, desc="Ffill, Bfill, Zeros Imputation",
                                           total=len(training_files)):
                example = pd.read_csv(training_file, sep=',')
                example.ffill(inplace=True)
                example.bfill(inplace=True)
                example.fillna(value=0, inplace=True)
                # example.fillna(means, inplace=True)
                training_examples.append(example)

                example.to_csv(os.path.join(self.destination_path, training_file), index=False)
            with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
                pickle.dump(training_examples, f)
            
            print(f"fill_missing_values() -> Dataset is saved under the name: {dataset_name}")

        return dataset_name

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

        # All added features
        added_features = ['MAP_SOFA', 'Bilirubin_total_SOFA', 'Platelets_SOFA', 'SOFA_score', 'SOFA_score_diff',
                          'SOFA_deterioration', 'ResP_qSOFA', 'SBP_qSOFA', 'qSOFA_score', 'qSOFA_score_diff',
                          'qSOFA_deterioration', 'qSOFA_indicator', 'SOFA_indicator', 'Mortality_sofa',
                          'Temp_sirs', 'HR_sirs', 'Resp_sirs', 'paco2_sirs', 'wbc_sirs', 'infection_proxy',
                          't_suspicion', 't_sofa', 't_sepsis']

        # Removing t_suspicion, t_sofa, and t_sepsis, infection_proxy
        # added_features = ['MAP_SOFA', 'Bilirubin_total_SOFA', 'Platelets_SOFA', 'SOFA_score', 'SOFA_score_diff',
        #                   'SOFA_deterioration', 'ResP_qSOFA', 'SBP_qSOFA', 'qSOFA_score', 'qSOFA_score_diff',
        #                   'qSOFA_deterioration', 'qSOFA_indicator', 'SOFA_indicator', 'Mortality_sofa',
        #                   'Temp_sirs', 'HR_sirs', 'Resp_sirs', 'paco2_sirs', 'wbc_sirs']

        with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
            pickle.dump(training_examples, f)

        print(f"add_additional_features() -> Dataset is saved under the name: {dataset_name}")

        return dataset_name, added_features

    def scale_features(self):
        dataset_name = 'final_dataset.pickle'

        # Fit using all the data
        data_path = os.path.join(os.path.join(project_root(), 'data', 'processed', 'final_dataset.pickle'))
        all_data = pd.read_pickle(data_path)
        data_concat = pd.concat(all_data)

        # Columns
        columns_to_scale = data_concat.drop(['SepsisLabel', 'PatientID'], axis=1).columns

        scaler = MinMaxScaler()
        scaler.fit(data_concat[columns_to_scale])

        # Transform patient-by-patient
        training_examples = []
        for example in tqdm.tqdm(all_data, desc="Scaling Features", total=len(all_data)):
            example[columns_to_scale] = pd.DataFrame(scaler.transform(example[columns_to_scale]))
            training_examples.append(example)

        with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
            pickle.dump(training_examples, f)
        
        print(f"scale_features() -> Dataset is saved under the name: {dataset_name}")

        return dataset_name


    def remove_unwanted_features(self, dataset_name, case_num, additional_features):
        print(f"remove_unwanted_features() -> Using dataset_name: {dataset_name}")

        vital_signs, laboratory_values, demographics = get_features(case=case_num)
        additional_features = ['SepsisLabel', 'PatientID'] + additional_features
        final_features = vital_signs + laboratory_values + demographics + additional_features

        print(f"Total number of features: {len(final_features)}")

        dataset = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', dataset_name))
        for idx, patient_df in tqdm.tqdm(enumerate(dataset), desc="Removing unwanted features", total=len(dataset)):
            dataset[idx] = patient_df[final_features]

        filtered_dataset_path = os.path.join(project_root(), 'data', 'processed', 'final_dataset.pickle')
        with open(filtered_dataset_path, 'wb') as file:
            pickle.dump(dataset, file)

        print(f"remove_unwanted_features() -> Dataset is saved under the name: {dataset_name}")

    def extract_timesteps(self, data, sepsis):
        if sepsis[0]:
            sepsis_index = data.index[data['SepsisLabel'] == 1][0]
            start_index = max(0, sepsis_index - 5)
            end_index = min(len(data), sepsis_index + 9)
            extracted_data = data.iloc[start_index:end_index]
        else:
            extracted_data = data.sample(n=14)

        return extracted_data

    def save_filtered_data(self, dataset, is_sepsis):

        dataset_name = 'final_dataset.pickle'
        assert len(dataset) == len(is_sepsis), f"{len(dataset)} != {len(is_sepsis)}"
        print(len(dataset), len(is_sepsis))
        filtered_data, filtered_sepsis = [], []
        for data, sepsis in tqdm.tqdm(zip(dataset, is_sepsis), desc="Extracting Time Steps: ", total=len(dataset)):
            if len(data) >= 14:
                filtered_data.append(self.extract_timesteps(data, sepsis))
                filtered_sepsis.append(sepsis[0])

        print(f"Total number of samples after filtering: {len(filtered_data)} and {len(filtered_sepsis)}")

        with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
            pickle.dump(filtered_data, f)

        with open(os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt'), 'w') as f:
            [f.write(f'{l}\n') for l in filtered_sepsis]



if __name__ == '__main__':

    setup = DataSetup()

    # Converts psv to csv
    setup.convert_to_csv()

    # Rewriting data
    csv_path = os.path.join(project_root(), 'data', 'csv')
    training_files = [os.path.join(csv_path, f) for f in os.listdir(csv_path) if f.endswith('.csv')]
    training_files.sort()
    setup.rewrite_csv(training_files=training_files)

    # Filling missing values and save csv files back
    dataset_name = setup.fill_missing_values(method='ffill_bfill', training_files=training_files)

    # Add features
    dataset = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', dataset_name))
    dataset_name, added_features = setup.add_additional_features(data=dataset)

    # Filtering (14 timesteps)
    dataset = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', dataset_name))
    is_sepsis = pd.read_csv(os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt'), header=None).values
    setup.save_filtered_data(dataset, is_sepsis)

    # Scaling features
    dataset = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', dataset_name))
    dataset_name = setup.scale_features()

    # Remove unwanted features
    # setup.remove_unwanted_features(case_num=1, additional_features=added_features, dataset_name=dataset_name)
