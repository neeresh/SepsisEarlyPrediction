import pandas as pd
import numpy as np

# Define optimized versions of the feature functions from add_features.py
def map_sofa(map_values):
    return np.where(map_values >= 70, 0, 1)

def platelets_sofa(platelets):
    return np.select(
        [platelets > 150, platelets >= 101, platelets >= 51, platelets >= 21],
        [0, 1, 2, 3],
        default=4
    )

def total_bilirubin_sofa(bilirubin):
    return np.select(
        [bilirubin < 1.2, bilirubin <= 1.9, bilirubin <= 5.9, bilirubin <= 11.9],
        [0, 1, 2, 3],
        default=4
    )

def respiratory_rate_qsofa(respiratory_rate):
    return (respiratory_rate >= 22).astype(int)

def sbp_qsofa(sbp):
    return (sbp < 100).astype(int)

def sofa_score(row):
    return row['Platelets_SOFA'] + row['Bilirubin_total_SOFA'] + row['MAP_SOFA']

def detect_sofa_change(data, time_window=24):
    data['SOFA_score_diff'] = data['SOFA_score'].diff(periods=time_window).fillna(0)
    data['SOFA_deterioration'] = (data['SOFA_score_diff'] >= 2).astype(int)
    return data

def qsofa_score(row):
    return row['ResP_qSOFA'] + row['SBP_qSOFA']
