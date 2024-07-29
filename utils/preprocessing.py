import numpy as np
import pandas as pd

from utils.add_features import *
from utils.evaluate_helper_methods import t_suspicion
from itertools import chain


def feature_informative_missingness(case, sep_columns):

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


def add_feature_informative_missingness(patient_data):

    vital_signs = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
    laboratory_values = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
                         'Calcium', 'Chloride', 'Creatinine', 'Glucose', 'Lactate',
                         'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'Hct', 'Hgb',
                         'PTT', 'WBC', 'Platelets']

    patient_data = feature_informative_missingness(patient_data, vital_signs + laboratory_values)

    return patient_data


def fill_missing_values(patient_data):
    patient_data.ffill(inplace=True)
    patient_data.bfill(inplace=True)
    patient_data.fillna(value=0, inplace=True)

    return patient_data


def feature_slide_window(temp, con_index):
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


def add_sliding_features_for_vital_signs(patient_data):
    # Ignoring temp, dbp and ETCO2  because of their missing values
    vital_signs = ['HR', 'O2Sat', 'SBP', 'MAP', 'Resp']
    vital_signs_idxs = [0, 1, 3, 4, 6]

    stats = ['max', 'min', 'mean', 'median', 'std', 'diff_std']

    # New columns names
    new_columns = []
    for col in vital_signs:
        for stat in stats:
            new_columns.append(f"{col}_{stat}")

    # example = pd.read_csv(training_file, sep=',')
    example = patient_data
    patient_data = feature_slide_window(patient_data.values, vital_signs_idxs)
    patient_data = pd.DataFrame(patient_data, columns=new_columns)

    patient_data = pd.concat([example, patient_data], axis=1)

    return patient_data


def add_additional_features(patient_data):
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

    return patient_data


def preprocessing(patient_data):
    patient_data = add_feature_informative_missingness(patient_data)
    patient_data = fill_missing_values(patient_data)
    patient_data = add_sliding_features_for_vital_signs(patient_data)
    patient_data = add_additional_features(patient_data)

    return patient_data


# def pad_rows(patient_data):
#
#     max_rows = 336
#     num_features = patient_data.shape[1]
#     if len(patient_data) < max_rows:
#         padding = np.zeros((max_rows - len(patient_data), num_features))
#         patient_data = np.vstack((patient_data.values, padding)).astype(np.float32)
#     else:
#         patient_data = patient_data.values.astype(np.float32)
#
#     return patient_data

def pad_rows(patient_data):
    max_rows = 336
    num_features = patient_data.shape[1]

    if len(patient_data) < max_rows:
        padding = np.zeros((max_rows - len(patient_data), num_features))
        patient_data_padded = np.vstack((patient_data.values, padding)).astype(np.float32)

        # Creating the mask
        mask = np.ones((max_rows, num_features), dtype=bool)
        mask[len(patient_data):, :] = False  # Mark padded rows as False
    else:
        patient_data_padded = patient_data.values.astype(np.float32)
        mask = np.ones_like(patient_data_padded, dtype=bool)  # No padding, mask all True

    return patient_data_padded, mask

