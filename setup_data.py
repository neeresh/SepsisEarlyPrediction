from itertools import chain

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

        self.data_paths = [os.path.join(project_root(), 'physionet.org', 'files', 'challenge-2019', '1.0.0', 'training',
                                        'training_setA'),
                           # os.path.join(project_root(), 'physionet.org', 'files', 'challenge-2019', '1.0.0', 'training',
                           #              'training_setB')
                           ]

        self.destination_path = os.path.join(project_root(), 'data', 'csv')
        self.hdf_path = os.path.join(project_root(), 'data', 'hdf')
        self.csv_path = os.path.join(project_root(), 'data', 'csv')

    def convert_to_csv(self):
        i = 0
        for data_path in self.data_paths:
            training_files = [file for file in os.listdir(data_path) if file.endswith('.psv')]
            training_files.sort()

            for _, file in enumerate(
                    tqdm.tqdm(training_files, desc="Converting psv to csv", total=len(training_files))):
                try:
                    temp = pd.read_csv(os.path.join(data_path, file), sep='|')
                    temp['PatientID'] = i
                    temp.to_csv(os.path.join(self.destination_path, file.replace('.psv', '.csv')), sep=',', index=False)
                    i += 1
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

        print("Conversion complete.")

    def rewrite_csv(self, training_files):

        training_examples, lengths, is_sepsis = [], [], []
        for i, training_file in enumerate(tqdm.tqdm(training_files, desc="Creating lengths_list, and is_sepsis files",
                                                    total=len(training_files))):

            example = pd.read_csv(training_file, sep=',')
            training_examples.append(example)
            is_sepsis.append(1 if 1 in example['SepsisLabel'].values else 0)
            lengths.append(len(example))

        with open(os.path.join(project_root(), 'data', 'processed', 'lengths.txt'), 'w') as f:
            [f.write(f'{l}\n') for l in lengths]

        with open(os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt'), 'w') as f:
            [f.write(f'{l}\n') for l in is_sepsis]

        # with open(os.path.join(project_root(), 'data', 'processed', 'final_dataset.pickle'), 'wb') as f:
        #     pickle.dump(training_examples, f)

    # FiO2 should be percentage between 0 and 1
    def preprocess_fio2(self, fio2):
        return np.select(
            condlist=[fio2 > 1.0, fio2 <= 1.0],
            choicelist=[fio2 / 100.0, fio2],
            default=np.nan
        )

    def fill_missing_values(self, pickle_file, method):

        training_files = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', pickle_file))
        # training_files = pickle_file

        if method == 'mean':
            all_data = pd.concat(training_files)

            dataset_name = 'training_mean.pickle'
            print(f"Filling missing values with mean")
            training_examples = []
            means = all_data.mean(axis=0, skipna=True)
            for training_file in tqdm.tqdm(training_files, desc="Mean Imputation", total=len(training_files)):
                example = pd.read_csv(training_file, sep=',')
                example['FiO2'] = self.preprocess_fio2(example['FiO2'])
                example.fillna(means, inplace=True)
                training_examples.append(example)

                example.to_csv(os.path.join(self.destination_path, training_file), index=False)
            with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
                pickle.dump(training_examples, f)

            print(f"fill_missing_values() -> Dataset is saved under the name: {dataset_name}")

        elif method == 'median':
            all_data = pd.concat(training_files)

            dataset_name = 'training_median.pickle'
            print(f"Filling missing values with median")
            training_examples = []
            medians = all_data.median(axis=0, skipna=True)
            for training_file in tqdm.tqdm(training_files, desc="Median Imputation", total=len(training_files)):
                example = pd.read_csv(training_file, sep=',')
                example['FiO2'] = self.preprocess_fio2(example['FiO2'])
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
                example = example.drop(['SepsisLabel'], axis=1)
                example['FiO2'] = self.preprocess_fio2(example['FiO2'])
                example.fillna(0, inplace=True)
                training_examples.append(example)

                example.to_csv(os.path.join(self.destination_path, training_file), index=False)
            with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
                pickle.dump(training_examples, f)

            print(f"fill_missing_values() -> Dataset is saved under the name: {dataset_name}")

        elif method == 'rolling':
            all_data = pd.concat(training_files)

            dataset_name = 'training_rolling.pickle'
            print(f"Filling vital featuers using rolling method")
            print(f"Filling all the other values with the ffill and bfill method")

            # vital_signs = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
            vital_features, _, _ = get_features(case=1)

            means = all_data.mean(axis=0, skipna=True)
            training_examples = []
            for training_file in tqdm.tqdm(training_files, desc="Filling missing using rolling & bfill(), and means",
                                           total=len(training_files)):
                example = pd.read_csv(training_file, sep=',')
                example['FiO2'] = self.preprocess_fio2(example['FiO2'])
                for feature in vital_features:
                    example[feature] = example[feature].rolling(window=6, min_periods=1).mean().bfill()
                example.fillna(means, inplace=True)
                training_examples.append(example)

                example.to_csv(os.path.join(self.destination_path, training_file), index=False)

            with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
                pickle.dump(training_examples, f)

            print(f"fill_missing_values() -> Dataset is saved under the name: {dataset_name}")

        elif method == 'custom_fill':
            mean_imputation = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'FiO2', 'pH', 'PaCO2',
                               'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Creatinine', 'Bilirubin_direct', 'Glucose',
                               'Lactate', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'Fibrinogen', 'Platelets',
                               'Age']
            median_imputation = ['HCO3', 'Chloride', 'Magnesium', 'Phosphate', 'Potassium', 'PTT', 'WBC',
                                 'HospAdmTime', 'ICULOS']
            category_imputation = ['Gender', 'Unit1', 'Unit2']

            all_data = pd.concat(training_files)
            means = all_data[mean_imputation].mean(axis=0, skipna=True).round(2)
            medians = all_data[median_imputation].median(axis=0, skipna=True).round(2)

            means.to_csv('means.csv', index=mean_imputation)
            medians.to_csv('medians.csv', index=median_imputation)

            training_examples = []
            print(f"Filling values using custom methods")
            for training_file in tqdm.tqdm(training_files, desc="Custom Imputation", total=len(training_files)):
                example = training_file
                example['FiO2'] = self.preprocess_fio2(example['FiO2'])

                example[mean_imputation] = example[mean_imputation].fillna(means)
                example[median_imputation] = example[median_imputation].fillna(medians)

                example['Unit1'] = example['Unit1'].fillna(0)
                example['Unit2'] = example['Unit2'].fillna(0)

                example.loc[example['Unit1'] == 0, 'Unit2'] = 1
                example.loc[example['Unit2'] == 0, 'Unit1'] = 1

                example.ffill(inplace=True)
                example.bfill(inplace=True)
                example.fillna(value=0, inplace=True)

                training_examples.append(example)

            dataset_name = 'final_dataset.pickle'
            with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
                pickle.dump(training_examples, f)

            print(f"custom_fill() -> Dataset is saved under the name: {dataset_name}")

        else:
            dataset_name = 'training_ffill_bfill_zeros.pickle'
            print(f"Filling missing values with ffill, bfill, and zeros")
            training_examples = []
            for training_file in tqdm.tqdm(training_files, desc="Ffill, Bfill, Zeros Imputation",
                                           total=len(training_files)):
                # example = pd.read_csv(training_file, sep=',')
                example = training_file
                example['FiO2'] = self.preprocess_fio2(example['FiO2'])
                example.ffill(inplace=True)
                example.bfill(inplace=True)
                example.fillna(value=0, inplace=True)
                training_examples.append(example)

                # example.to_csv(os.path.join(self.destination_path, training_file), index=False)
            with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
                pickle.dump(training_examples, f)

            print(f"fill_missing_values() -> Dataset is saved under the name: {dataset_name}")

        return dataset_name

    def add_additional_features(self, pickle_file):

        data = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', pickle_file))

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

            # NEWS - National Early Warning Score
            patient_data['HR_NEWS'] = hr_news(patient_data['HR'])
            patient_data['Temp_NEWS'] = temp_news(patient_data['Temp'])
            patient_data['Resp_NEWS'] = resp_news(patient_data['Resp'])
            patient_data['Creatinine_NEWS'] = creatinine_news(patient_data['Creatinine'])
            patient_data['MAP_NEWS'] = map_news(patient_data['MAP'])

            training_examples.append(patient_data)

        # All added features
        added_features = ['MAP_SOFA', 'Bilirubin_total_SOFA', 'Platelets_SOFA', 'SOFA_score', 'SOFA_score_diff',
                          'SOFA_deterioration', 'ResP_qSOFA', 'SBP_qSOFA', 'qSOFA_score', 'qSOFA_score_diff',
                          'qSOFA_deterioration', 'qSOFA_indicator', 'SOFA_indicator', 'Mortality_sofa',
                          'Temp_sirs', 'HR_sirs', 'Resp_sirs', 'paco2_sirs', 'wbc_sirs', 'infection_proxy',
                          't_suspicion', 't_sofa', 't_sepsis', 'HR_NEWS', 'Temp_NEWS', 'Resp_NEWS', 'Creatinine_NEWS',
                          'MAP_NEWS']

        # Removing t_suspicion, t_sofa, and t_sepsis, infection_proxy
        # added_features = ['MAP_SOFA', 'Bilirubin_total_SOFA', 'Platelets_SOFA', 'SOFA_score', 'SOFA_score_diff',
        #                   'SOFA_deterioration', 'ResP_qSOFA', 'SBP_qSOFA', 'qSOFA_score', 'qSOFA_score_diff',
        #                   'qSOFA_deterioration', 'qSOFA_indicator', 'SOFA_indicator', 'Mortality_sofa',
        #                   'Temp_sirs', 'HR_sirs', 'Resp_sirs', 'paco2_sirs', 'wbc_sirs']

        saved_as = "final_dataset.pickle"
        with open(os.path.join(project_root(), 'data', 'processed', saved_as), 'wb') as f:
            pickle.dump(training_examples, f)

        print(f"add_additional_features() -> Dataset is saved under the name: {saved_as}")

        return saved_as, added_features

    def scale_features(self, pickle_file):

        dataset = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', pickle_file))

        # Fit using all the data
        # data_path = os.path.join(os.path.join(project_root(), 'data', 'processed', dataset))
        # all_data = pd.read_pickle(data_path)
        data_concat = pd.concat(dataset)

        # Columns
        columns_to_scale = data_concat.drop(['SepsisLabel', 'PatientID'], axis=1).columns

        scaler = StandardScaler()
        scaler.fit(data_concat[columns_to_scale])

        # Transform patient-by-patient
        training_examples = []
        for example in tqdm.tqdm(dataset, desc="Scaling Features", total=len(dataset)):
            example[columns_to_scale] = pd.DataFrame(scaler.transform(example[columns_to_scale]))
            training_examples.append(example)

        saved_as = 'final_dataset.pickle'
        with open(os.path.join(project_root(), 'data', 'processed', saved_as), 'wb') as f:
            pickle.dump(training_examples, f)

        print(f"scale_features() -> Dataset is saved under the name: {saved_as}")

        return saved_as

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

    def feature_slide_window(self, temp, con_index):

        sepdata = temp[:, con_index]
        max_values = [[0 for col in range(len(sepdata))]
                      for row in range(len(con_index))]
        min_values = [[0 for col in range(len(sepdata))]
                      for row in range(len(con_index))]
        mean_values = [[0 for col in range(len(sepdata))]
                       for row in range(len(con_index))]
        median_values = [[0 for col in range(len(sepdata))]
                         for row in range(len(con_index))]
        std_values = [[0 for col in range(len(sepdata))]
                      for row in range(len(con_index))]
        diff_std_values = [[0 for col in range(len(sepdata))]
                           for row in range(len(con_index))]

        for i in range(len(sepdata)):
            if i < 6:
                win_data = sepdata[0:i + 1]
                for ii in range(6 - i):
                    win_data = np.row_stack((win_data, sepdata[i]))
            else:
                win_data = sepdata[i - 6: i + 1]

            for j in range(len(con_index)):
                dat = win_data[:, j]
                if len(np.where(~np.isnan(dat))[0]) == 0:
                    max_values[j][i] = np.nan
                    min_values[j][i] = np.nan
                    mean_values[j][i] = np.nan
                    median_values[j][i] = np.nan
                    std_values[j][i] = np.nan
                    diff_std_values[j][i] = np.nan
                else:
                    max_values[j][i] = np.nanmax(dat)
                    min_values[j][i] = np.nanmin(dat)
                    mean_values[j][i] = np.nanmean(dat)
                    median_values[j][i] = np.nanmedian(dat)
                    std_values[j][i] = np.nanstd(dat)
                    diff_std_values[j][i] = np.std(np.diff(dat))

        win_features = list(chain(max_values, min_values, mean_values,
                                  median_values, std_values, diff_std_values))
        win_features = (np.array(win_features)).T

        return win_features

    def add_sliding_features_for_vital_signs(self, pickle_file):

        training_files = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', pickle_file))

        # Ignoring temp, dbp and ETCO2  because of their missing values
        vital_signs = ['HR', 'O2Sat', 'SBP', 'MAP', 'Resp']
        vital_signs_idxs = [0, 1, 3, 4, 6]

        stats = ['max', 'min', 'mean', 'median', 'std', 'diff_std']

        # New columns names
        new_columns = []
        for col in vital_signs:
            for stat in stats:
                new_columns.append(f"{col}_{stat}")

        assert 6 * len(vital_signs) == len(new_columns), (
            f"Incorrect number of columns created: {6 * len(vital_signs)} "
            f"and {len(new_columns)}")

        training_examples = []
        for training_file in tqdm.tqdm(training_files, desc="Creating vital window features",
                                       total=len(training_files)):
            # example = pd.read_csv(training_file, sep=',')
            example = training_file
            temp_example = self.feature_slide_window(example.values, vital_signs_idxs)
            temp_example = pd.DataFrame(temp_example, columns=new_columns)

            training_examples.append(pd.concat([example, temp_example], axis=1))

        saved_as = "final_dataset.pickle"
        with open(os.path.join(project_root(), 'data', 'processed', saved_as), 'wb') as f:
            pickle.dump(training_examples, f)

        print(f"add_sliding_features_for_vital_signs() -> Dataset is saved under the name: {saved_as}")

        return saved_as

    def feature_informative_missingness(self, case, sep_columns):

        temp_data = case.to_numpy()
        column_names = list(case.columns)

        for sep_column in sep_columns:
            sep_data = case[sep_column].to_numpy()
            nan_pos = np.where(~np.isnan(sep_data))[0]

            # Measurement frequency sequence
            interval_f1 = sep_data.copy()
            interval_f1_name = f"interval_f1_{sep_column}"
            if len(nan_pos) == 0:
                interval_f1[:] = 0
            else:
                interval_f1[: nan_pos[0]] = 0
                for p in range(len(nan_pos) - 1):
                    interval_f1[nan_pos[p]: nan_pos[p + 1]] = p + 1
                interval_f1[nan_pos[-1]:] = len(nan_pos)

            temp_data = np.column_stack((temp_data, interval_f1))
            column_names.append(interval_f1_name)

            # Measurement time interval
            interval_f2 = sep_data.copy()
            interval_f2_name = f"interval_f2_{sep_column}"
            if len(nan_pos) == 0:
                interval_f2[:] = -1
            else:
                interval_f2[:nan_pos[0]] = -1
                for q in range(len(nan_pos) - 1):
                    length = nan_pos[q + 1] - nan_pos[q]
                    for l in range(length):
                        interval_f2[nan_pos[q] + l] = l

                length = len(case) - nan_pos[-1]
                for l in range(length):
                    interval_f2[nan_pos[-1] + l] = l

            temp_data = np.column_stack((temp_data, interval_f2))
            column_names.append(interval_f2_name)

            # Differential features
            diff_f = sep_data.copy()
            diff_f = diff_f.astype(float)
            diff_f_name = f"diff_f_{sep_column}"
            if len(nan_pos) <= 1:
                diff_f[:] = np.NaN
            else:
                diff_f[:nan_pos[1]] = np.NaN
                for p in range(1, len(nan_pos) - 1):
                    diff_f[nan_pos[p]: nan_pos[p + 1]] = sep_data[nan_pos[p]] - sep_data[nan_pos[p - 1]]
                diff_f[nan_pos[-1]:] = sep_data[nan_pos[-1]] - sep_data[nan_pos[-2]]

            temp_data = np.column_stack((temp_data, diff_f))
            column_names.append(diff_f_name)

        patient_data = pd.DataFrame(temp_data, columns=column_names)

        return patient_data

    def add_feature_informative_missingness(self, training_files):

        # 99% of the missing features are ignored here.
        # features = ['Bilirubin_direct', 'TroponinI', 'Fibrinogen']
        vital_signs = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
        laboratory_values = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
                             'Calcium', 'Chloride', 'Creatinine', 'Glucose', 'Lactate',
                             'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'Hct', 'Hgb',
                             'PTT', 'WBC', 'Platelets']

        training_examples = []
        for training_file in tqdm.tqdm(training_files, desc="Adding Feature Informative Missing-ness",
                                       total=len(training_files)):
            training_file = pd.read_csv(training_file)
            sepsis_label = training_file['SepsisLabel']
            added_missingness = self.feature_informative_missingness(training_file.drop(['SepsisLabel'], axis=1),
                                                                     vital_signs + laboratory_values)

            training_file = pd.concat([added_missingness, sepsis_label], axis=1)
            training_examples.append(training_file)

        save_as = "final_dataset.pickle"
        with open(os.path.join(project_root(), 'data', 'processed', save_as), 'wb') as f:
            pickle.dump(training_examples, f)

        print(f"add_feature_informative_missingness() -> Dataset is saved under the name: {save_as}")

        return save_as

    def convert_csv_to_psv(self, pickle_file):
        """
        For testing
        """

        # To test only training_setA, edit evaluation code files to [:20336]
        training_files = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', pickle_file))

        csv_path = os.path.join(project_root(), 'data', 'csv')
        file_names = [os.path.join(csv_path, f) for f in os.listdir(csv_path) if f.endswith('.csv')]
        file_names.sort()

        destination_path = os.path.join(project_root(), 'data', 'test_data')

        for i, (patient_data, file_name) in enumerate(
                tqdm.tqdm(zip(training_files, file_names), desc="Converting CSV to PSV",
                          total=len(training_files))):
            patient_data = patient_data.drop(['PatientID', 'SepsisLabel'], axis=1)
            psv_file_name = file_name.split('/')[-1].replace('.csv', '.psv')
            patient_data.to_csv(os.path.join(destination_path, psv_file_name), sep='|', index=False)


if __name__ == '__main__':
    setup = DataSetup()

    # Converts psv to csv; Output: All psv files to csv files
    setup.convert_to_csv()

    # Rewriting data; Output: lengths.txt, is_sepsis.txt
    csv_path = os.path.join(project_root(), 'data', 'csv')
    training_files = [os.path.join(csv_path, f) for f in os.listdir(csv_path) if f.endswith('.csv')]
    training_files.sort()
    setup.rewrite_csv(training_files=training_files)

    # Add Feature Informative Missing-ness; Output: final_dataset.pickle
    final_dataset_pickle = setup.add_feature_informative_missingness(training_files=training_files)

    # Filling missing values and save csv files back; Output: final_dataset.pickle
    saved_as = setup.fill_missing_values(pickle_file='final_dataset.pickle', method='custom_fill')

    # Sliding window features for vital signs; Output: final_dataset.pickle
    saved_as = setup.add_sliding_features_for_vital_signs(pickle_file='final_dataset.pickle')

    # Add features - Scores
    # Output: final_dataset.pickle
    saved_as, added_features = setup.add_additional_features(pickle_file='final_dataset.pickle')

    # # Filtering (14 timesteps)
    # dataset = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', dataset_name))
    # is_sepsis = pd.read_csv(os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt'), header=None).values
    # setup.save_filtered_data(dataset, is_sepsis)

    # Scaling features
    # Output: final_dataset.pickle
    saved_as = setup.scale_features(pickle_file="final_dataset.pickle")

    # # Remove unwanted features
    # # setup.remove_unwanted_features(case_num=1, additional_features=added_features, dataset_name=dataset_name)

    # # Convert all files to psv in test folder (for evaluation)
    # setup.convert_csv_to_psv(pickle_file='final_dataset.pickle')
