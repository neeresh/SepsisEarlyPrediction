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


# FiO2 should be percentage between 0 and 1
def preprocess_fio2(fio2):
    return np.select(
        condlist=[fio2 > 1.0, fio2 <= 1.0],
        choicelist=[fio2 / 100.0, fio2],
        default=np.nan
    )


class DataSetup:
    def __init__(self):
        self.data_paths = [
            os.path.join(project_root(), 'physionet.org', 'files', 'challenge-2019', '1.0.0', 'training',
                         'training_setA'),
            # os.path.join(project_root(), 'physionet.org', 'files', 'challenge-2019', '1.0.0', 'training',
            #              'training_setB')
        ]

        self.destination_path = os.path.join(project_root(), 'data', 'csv')
        self.csv_path = os.path.join(project_root(), 'data', 'csv')

        self.lengths_path = os.path.join(project_root(), 'data', 'processed', 'lengths.txt')
        self.sepsis_path = os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt')

        self.pickle_path = os.path.join(project_root(), 'data', 'processed')

    def convert_to_csv(self):
        i = 0
        for data_path in self.data_paths:
            all_samples = [file for file in os.listdir(data_path) if file.endswith('.psv')]
            all_samples.sort()

            for _, file in enumerate(
                    tqdm.tqdm(all_samples, desc="Converting psv to csv...", total=len(all_samples))):
                try:
                    temp = pd.read_csv(os.path.join(data_path, file), sep='|')
                    temp['PatientID'] = i
                    temp.to_csv(os.path.join(self.destination_path, file.replace('.psv', '.csv')), sep=',',
                                index=False)
                    i += 1
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

        print("Conversion complete.")

    def rewrite_csv(self, all_samples):

        training_examples, lengths, is_sepsis = [], [], []
        for i, training_file in enumerate(tqdm.tqdm(all_samples, desc="Creating lengths_list, and is_sepsis files",
                                                    total=len(all_samples))):
            example = pd.read_csv(training_file, sep=',')
            training_examples.append(example)
            is_sepsis.append(1 if 1 in example['SepsisLabel'].values else 0)
            lengths.append(len(example))

        with open(self.lengths_path, 'w') as f:
            [f.write(f'{length}\n') for length in lengths]

        with open(self.sepsis_path, 'w') as f:
            [f.write(f'{sepsis}\n') for sepsis in is_sepsis]

    def fill_missing_values(self, pickle_file, method):

        train_files = pd.read_pickle(self.pickle_path + f'/{pickle_file}')

        if method == 'rolling':
            all_data = pd.concat(train_files)

            dataset_name = 'training_rolling.pickle'
            print(f"Filling vital features using rolling method")
            print(f"Filling all the other features with the ffill and bfill")

            vital_features, _, _ = get_features(case=1)

            means = all_data.mean(axis=0, skipna=True)
            training_examples = []
            for training_file in tqdm.tqdm(training_files, desc="Filling missing using rolling & bfill(), and means",
                                           total=len(training_files)):
                example = pd.read_csv(training_file, sep=',')
                example['FiO2'] = preprocess_fio2(example['FiO2'])
                for feature in vital_features:
                    example[feature] = example[feature].rolling(window=6, min_periods=1).mean().bfill()
                example.fillna(means, inplace=True)
                training_examples.append(example)

                example.to_csv(os.path.join(self.destination_path, training_file), index=False)

            with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
                pickle.dump(training_examples, f)

            print(f"fill_missing_values() -> Dataset is saved under the name: {dataset_name}")

        elif method == 'custom_fill':

            # Removed Features: ['Bilirubin_direct' (mean), 'TroponinI' (mean), 'Fibrinogen' (mean)]
            mean_imputation = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'FiO2', 'pH', 'PaCO2',
                               'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Creatinine', 'Glucose',
                               'Lactate', 'Bilirubin_total', 'Hct', 'Hgb', 'Platelets',
                               'Age']
            median_imputation = ['HCO3', 'Chloride', 'Magnesium', 'Phosphate', 'Potassium', 'PTT', 'WBC',
                                 'HospAdmTime', 'ICULOS']
            category_imputation = ['Gender', 'Unit1', 'Unit2']

            all_data = pd.concat(train_files)
            means = all_data[mean_imputation].mean(axis=0, skipna=True).round(2)
            medians = all_data[median_imputation].median(axis=0, skipna=True).round(2)

            means.to_csv('means.csv', index=mean_imputation)
            medians.to_csv('medians.csv', index=median_imputation)

            training_examples = []
            print(f"Filling values using custom methods (mean, median, ffill, bfill, zeros)")

            for example in tqdm.tqdm(train_files, desc="Custom Imputation", total=len(training_files)):
                example['FiO2'] = preprocess_fio2(example['FiO2'])

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
            for example in tqdm.tqdm(train_files, desc="Ffill, Bfill, Zeros Imputation", total=len(training_files)):
                example['FiO2'] = preprocess_fio2(example['FiO2'])
                example.ffill(inplace=True)
                example.bfill(inplace=True)
                example.fillna(value=0, inplace=True)
                training_examples.append(example)

            with open(os.path.join(project_root(), 'data', 'processed', dataset_name), 'wb') as f:
                pickle.dump(training_examples, f)

            print(f"fill_missing_values() -> Dataset is saved under the name: {dataset_name}")

        return dataset_name

    def add_additional_features(self, pickle_file):

        data = pd.read_pickle(self.pickle_path + f'/{pickle_file}')

        training_examples = []
        for patient_id, pdata in tqdm.tqdm(enumerate(data), desc="Adding additional features", total=len(data)):
            pdata['MAP_SOFA'] = pdata['MAP'].apply(map_sofa)
            pdata['Bilirubin_total_SOFA'] = pdata['Bilirubin_total'].apply(total_bilirubin_sofa)
            pdata['Platelets_SOFA'] = pdata['Platelets'].apply(platelets_sofa)
            pdata['SOFA_score'] = pdata.apply(sofa_score, axis=1)
            pdata = detect_sofa_change(pdata)

            pdata['ResP_qSOFA'] = pdata['Resp'].apply(respiratory_rate_qsofa)
            pdata['SBP_qSOFA'] = pdata['SBP'].apply(sbp_qsofa)
            pdata['qSOFA_score'] = pdata.apply(qsofa_score, axis=1)
            pdata = detect_qsofa_change(pdata)

            pdata['qSOFA_indicator'] = pdata.apply(q_sofa_indicator, axis=1)  # Sepsis detected
            pdata['SOFA_indicator'] = pdata.apply(sofa_indicator, axis=1)  # Organ Dysfunction occurred
            pdata['Mortality_sofa'] = pdata.apply(mortality_sofa, axis=1)  # Morality rate

            pdata['Temp_sirs'] = pdata['Temp'].apply(temp_sirs)
            pdata['HR_sirs'] = pdata['HR'].apply(heart_rate_sirs)
            pdata['Resp_sirs'] = pdata['Resp'].apply(resp_sirs)
            pdata['paco2_sirs'] = pdata['PaCO2'].apply(resp_sirs)
            pdata['wbc_sirs'] = pdata['WBC'].apply(wbc_sirs)

            pdata = t_suspicion(pdata)
            pdata = t_sofa(pdata)
            pdata['t_sepsis'] = pdata.apply(t_sepsis, axis=1)

            # NEWS - National Early Warning Score
            pdata['HR_NEWS'] = hr_news(pdata['HR'])
            pdata['Temp_NEWS'] = temp_news(pdata['Temp'])
            pdata['Resp_NEWS'] = resp_news(pdata['Resp'])
            pdata['Creatinine_NEWS'] = creatinine_news(pdata['Creatinine'])
            pdata['MAP_NEWS'] = map_news(pdata['MAP'])

            training_examples.append(pdata)

        added_features = ['MAP_SOFA', 'Bilirubin_total_SOFA', 'Platelets_SOFA', 'SOFA_score', 'SOFA_score_diff',
                          'SOFA_deterioration', 'ResP_qSOFA', 'SBP_qSOFA', 'qSOFA_score', 'qSOFA_score_diff',
                          'qSOFA_deterioration', 'qSOFA_indicator', 'SOFA_indicator', 'Mortality_sofa',
                          'Temp_sirs', 'HR_sirs', 'Resp_sirs', 'paco2_sirs', 'wbc_sirs', 'infection_proxy',
                          't_suspicion', 't_sofa', 't_sepsis', 'HR_NEWS', 'Temp_NEWS', 'Resp_NEWS', 'Creatinine_NEWS',
                          'MAP_NEWS']

        saved_as = "final_dataset.pickle"
        with open(os.path.join(project_root(), 'data', 'processed', saved_as), 'wb') as f:
            pickle.dump(training_examples, f)

        print(f"add_additional_features() -> Dataset is saved under the name: {saved_as}")

        return saved_as, added_features

    def scale_features(self, pickle_file):

        data = pd.read_pickle(self.pickle_path + f'/{pickle_file}')
        data_concat = pd.concat(data)

        # Columns
        columns_to_scale = data_concat.drop(['SepsisLabel', 'PatientID'], axis=1).columns

        scaler = StandardScaler()
        scaler.fit(data_concat[columns_to_scale])

        # Transform patient-by-patient
        training_examples = []
        for example in tqdm.tqdm(data, desc="Scaling Features", total=len(data)):
            example[columns_to_scale] = pd.DataFrame(scaler.transform(example[columns_to_scale]))
            training_examples.append(example)

        scaler_save_as = 'scaler.pkl'
        with open(os.path.join(project_root(), 'da', scaler_save_as), 'wb') as f:
            pickle.dump(scaler, f)

        print(f"scale_features() -> Scaler is saved under the name: {scaler_save_as}")

        save_as = 'final_dataset.pickle'
        with open(os.path.join(project_root(), 'data', 'processed', save_as), 'wb') as f:
            pickle.dump(training_examples, f)

        print(f"scale_features() -> Dataset is saved under the name: {save_as}")

        return save_as

    def remove_unwanted_features(self, dataset_name, drop_features):

        print(f"Total number of features to be removed: {len(drop_features)}")

        data = pd.read_pickle(self.pickle_path + f'/{dataset_name}')

        for idx, patient_df in tqdm.tqdm(enumerate(data), desc="Removing unwanted features", total=len(data)):
            data[idx] = patient_df.drop(columns=drop_features, axis=1)

        dataset_name = "final_dataset.pickle"
        filtered_dataset_path = os.path.join(project_root(), 'data', 'processed', dataset_name)
        with open(filtered_dataset_path, 'wb') as file:
            pickle.dump(data, file)

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

        data = pd.read_pickle(self.pickle_path + 'f/{pickle_file}')

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
        for example in tqdm.tqdm(data, desc="Creating vital window features", total=len(data)):
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

    def add_feature_informative_missingness(self, data):

        # 99% of the missing features are ignored here.
        # features = ['Bilirubin_direct', 'TroponinI', 'Fibrinogen']
        vital_signs = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
        laboratory_values = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
                             'Calcium', 'Chloride', 'Creatinine', 'Glucose', 'Lactate',
                             'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'Hct', 'Hgb',
                             'PTT', 'WBC', 'Platelets']

        training_examples = []
        for training_file in tqdm.tqdm(data, desc="Adding Feature Informative Missing-ness",
                                       total=len(data)):
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
        data = pd.read_pickle(self.pickle_path + f'/{pickle_file}')

        csv_path = os.path.join(project_root(), 'data', 'csv')
        file_names = [os.path.join(csv_path, f) for f in os.listdir(csv_path) if f.endswith('.csv')]
        file_names.sort()

        destination_path = os.path.join(project_root(), 'data', 'test_data')

        for i, (pdata, file_name) in enumerate(
                tqdm.tqdm(zip(data, file_names), desc="Converting CSV to PSV",
                          total=len(data))):
            pdata = pdata.drop(['PatientID', 'SepsisLabel'], axis=1)
            psv_file_name = file_name.split('/')[-1].replace('.csv', '.psv')
            pdata.to_csv(os.path.join(destination_path, psv_file_name), sep='|', index=False)

    def convert_pickle_to_csv(self, pickle_file, dir='pretrain'):

        data = pd.read_pickle(self.pickle_path + f'/{pickle_file}')

        # Just for file names
        csvpath = os.path.join(project_root(), 'data', 'csv')
        file_names = [os.path.join(csvpath, f) for f in os.listdir(csvpath) if f.endswith('.csv')]
        file_names.sort()

        destination_path = os.path.join(project_root(), 'data', 'tl_datasets', dir)

        for i, (pdata, file_name) in enumerate(
                tqdm.tqdm(zip(data, file_names), desc="Converting csv to .pt",
                          total=len(data))):
            pdata = pdata.drop(['PatientID', 'SepsisLabel'], axis=1)
            csv_file_name = file_name.split('/')[-1].replace('.csv', '.csv')
            pdata.to_csv(os.path.join(destination_path, csv_file_name), index=False)

    def create_pickle_debug(self, data):

        training_examples = []
        for training_file in tqdm.tqdm(data, desc="Creating a pickle file",
                                       total=len(data)):
            training_file = pd.read_csv(training_file)
            training_examples.append(training_file)

        save_as = "final_dataset.pickle"
        with open(os.path.join(project_root(), 'data', 'processed', save_as), 'wb') as f:
            pickle.dump(training_examples, f)

        print(f"create_pickle_debug() -> created pickle file: {save_as}")

        return save_as

    def update_csv_files(self, pickle_file):
        data = pd.read_pickle(self.pickle_path + f'/{pickle_file}')

        # Just for file names
        csvpath = os.path.join(project_root(), 'data', 'csv')
        file_names = [os.path.join(csvpath, f) for f in os.listdir(csvpath) if f.endswith('.csv')]
        file_names.sort()

        destination_path = csvpath

        for i, (pdata, file_name) in enumerate(
                tqdm.tqdm(zip(data, file_names), desc="Converting csv to .pt",
                          total=len(data))):
            pdata = pdata.drop(['PatientID', 'SepsisLabel'], axis=1)
            csv_file_name = file_name.split('/')[-1].replace('.csv', '.csv')
            pdata.to_csv(os.path.join(destination_path, csv_file_name), index=False)


if __name__ == '__main__':

    setup = DataSetup()

    # Converts psv to csv; Output: All psv files to csv files
    setup.convert_to_csv()

    # Rewriting data; Output: lengths.txt, is_sepsis.txt
    csv_path = os.path.join(project_root(), 'data', 'csv')
    training_files = [os.path.join(csv_path, f) for f in os.listdir(csv_path) if f.endswith('.csv')]
    training_files.sort()
    setup.rewrite_csv(training_files)

    # # Add Feature Informative Missing-ness; Output: final_dataset.pickle
    # final_dataset_pickle = setup.add_feature_informative_missingness(training_files=training_files)

    final_dataset_pickle = setup.create_pickle_debug(training_files)  # Just for debuging purposes

    # # Remove unwanted features
    # setup.remove_unwanted_features(dataset_name="final_dataset.pickle",
    #                                drop_features=['Bilirubin_direct', 'TroponinI', 'Fibrinogen'])

    # Filling missing values and save csv files back; Output: final_dataset.pickle
    saved_as = setup.fill_missing_values(pickle_file='final_dataset.pickle', method='custom_fill')

    # # Sliding window features for vital signs; Output: final_dataset.pickle
    # saved_as = setup.add_sliding_features_for_vital_signs(pickle_file='final_dataset.pickle')

    # # Add features - Scores
    # # Output: final_dataset.pickle
    # saved_as, added_features = setup.add_additional_features(pickle_file='final_dataset.pickle')

    # Filtering (14 timesteps)
    # dataset = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', dataset_name))
    # is_sepsis = pd.read_csv(os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt'), header=None).values
    # setup.save_filtered_data(dataset, is_sepsis)

    # Scaling features
    # Output: final_dataset.pickle
    # saved_as = setup.scale_features(pickle_file="final_dataset.pickle")

    # # Convert all files to psv in test folder (for evaluation)
    # setup.convert_csv_to_psv(pickle_file='final_dataset.pickle')

    # setup.convert_pickle_to_csv(pickle_file='final_dataset.pickle', dir='')

    # Updating A+B
    # setup.update_csv_files('final_dataset.pickle')
