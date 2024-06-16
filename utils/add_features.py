import os

import numpy as np
import pandas as pd

from utils.path_utils import project_root


# Defining optimized versions of the feature functions from add_features.py
def map_sofa(map_values):
    return np.where(map_values >= 70, 0, 1)


def platelets_sofa(platelets):
    return np.select(
        condlist=[platelets > 150, platelets >= 101, platelets >= 51, platelets >= 21],
        choicelist=[0, 1, 2, 3], default=4)


def total_bilirubin_sofa(bilirubin):
    return np.select(
        condlist=[bilirubin < 1.2, bilirubin <= 1.9, bilirubin <= 5.9, bilirubin <= 11.9],
        choicelist=[0, 1, 2, 3], default=4)


def map_sofa(map):
    return np.select(condlist=[map >= 70], choicelist=[0], default=1)


def sofa_score(row):
    return row['Platelets_SOFA'] + row['Bilirubin_total_SOFA'] + row['MAP_SOFA']


def detect_sofa_change(data, time_window=24):
    data['SOFA_score_diff'] = data['SOFA_score'].diff(periods=time_window).fillna(0)
    data['SOFA_deterioration'] = (data['SOFA_score_diff'] >= 2).astype(int)
    return data


def respiratory_rate_qsofa(respiratory_rate):
    # return (respiratory_rate >= 22).astype(int)
    return int(respiratory_rate >= 22)

def sbp_qsofa(sbp):
    return int(sbp < 100)


def qsofa_score(row):
    return row['ResP_qSOFA'] + row['SBP_qSOFA']


def q_sofa_indicator(row):
    return np.select(
        condlist=[(row['ResP_qSOFA'] > 0 and row['SBP_qSOFA'] > 0)],
        choicelist=[1], default=0
    )


def sofa_indicator(row):
    return np.select(
        condlist=[(row['Platelets_SOFA'] + row['Bilirubin_total_SOFA'] + row['MAP_SOFA'] > 2)],
        choicelist=[1], default=0
    )


def detect_qsofa_change(data, time_window=24):
    data['qSOFA_score_diff'] = data['qSOFA_score'].diff(periods=time_window).fillna(0)
    data['qSOFA_deterioration'] = (data['qSOFA_score_diff'] >= 2).astype(int)

    return data


def mortality_sofa(row):
    total_points = row['Platelets_SOFA'] + row['Bilirubin_total_SOFA'] + row['MAP_SOFA']
    return np.select(
        condlist=[total_points <= 1, total_points <= 9, total_points < 14],
        choicelist=[0, 0.30, 0.50], default=0.95
    )


def temp_sirs(temp):
    return np.select(
        condlist=[temp < 36, temp >= 38], choicelist=[1, 1], default=0
    )


def heart_rate_sirs(heart_rate):
    # return (heart_rate > 90).astype(int)
    return int(heart_rate > 90)


def resp_sirs(resp):
    return int(resp > 20)


def paco2_sirs(paco2):
    return int(paco2 < 32)


def wbc_sirs(wbc):
    return np.select(
        condlist=[wbc*1000 < 4000, wbc*1000 > 12000],
        choicelist=[1, 1], default=0
    )


def t_suspicion(patient_data):
    """
    Since we don't have information about IV antibiotics and blood cultures,
    we are is considering that patient have infection if any 2 SIRS criteria are met
    """
    patient_data['infection_proxy'] = (patient_data[['Temp_sirs', 'HR_sirs', 'Resp_sirs']].eq(1).sum(axis=1) >= 2).astype(int)

    # t_suspicion is the first hour of (ICULOS) where infection proxy is positive at time t
    patient_data['t_suspicion'] = patient_data.groupby(['PatientID'])['ICULOS'].transform(
        lambda x: x[patient_data['infection_proxy'] == 1].min() if (patient_data['infection_proxy'] == 1).any() else 0)

    return patient_data


def t_sofa(data):
    """
    Two-point deterioration in SOFA score at time t but within a 24-hour period.
    """
    data['t_sofa'] = data['SOFA_score_diff'].where((abs(data['SOFA_score_diff']) >= 2) & (data['ICULOS'] <= 24),
                                                   other=0)
    return data


def t_sepsis(row):
    if pd.isna(row['t_suspicion']) or row['t_suspicion'] == 0 or row['t_sofa'] == 0:
        return 0
    if row['t_suspicion'] - 24 <= row['t_sofa'] <= row['t_suspicion'] + 12:
        return min(row['t_suspicion'], row['t_sofa'])


if __name__ == '__main__':
    dataset = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', 'training_ffill_bfill_zeros.pickle'))
    for idx, patient_data in enumerate(dataset):
        if idx == 8:
            break

    patient_data['MAP_SOFA'] = map_sofa(patient_data['MAP'])
    patient_data['Bilirubin_total_SOFA'] = total_bilirubin_sofa(patient_data['Bilirubin_total'])
    patient_data['Platelets_SOFA'] = platelets_sofa(patient_data['Platelets'])
    patient_data['SOFA_score'] = patient_data[['MAP_SOFA', 'Bilirubin_total_SOFA', 'Platelets_SOFA']].sum(axis=1)
    patient_data = detect_sofa_change(patient_data)

    patient_data['ResP_qSOFA'] = respiratory_rate_qsofa(patient_data['Resp'])
    patient_data['SBP_qSOFA'] = sbp_qsofa(patient_data['SBP'])
    patient_data['qSOFA_score'] = patient_data[['ResP_qSOFA', 'SBP_qSOFA']].sum(axis=1)
    patient_data = detect_qsofa_change(patient_data)

    patient_data['qSOFA_indicator'] = patient_data.apply(q_sofa_indicator, axis=1)  # Sepsis detected
    patient_data['SOFA_indicator'] = patient_data.apply(sofa_indicator, axis=1)  # Organ Dysfunction occurred
    patient_data['Mortality_sofa'] = patient_data.apply(mortality_sofa, axis=1)  # Morality rate

    patient_data['Temp_sirs'] = temp_sirs(patient_data['Temp'])
    patient_data['HR_sirs'] = heart_rate_sirs(patient_data['HR'])
    patient_data['Resp_sirs'] = resp_sirs(patient_data['Resp'])
    patient_data['paco2_sirs'] = paco2_sirs(patient_data['PaCO2'])
    patient_data['wbc_sirs'] = wbc_sirs(patient_data['WBC'])

    patient_data = t_suspicion(patient_data)
    patient_data = t_sofa(patient_data)
    patient_data['t_sepsis'] = patient_data.apply(t_sepsis, axis=1)

    print(f"Total number of features: {patient_data.shape[1]}:", patient_data.columns)
