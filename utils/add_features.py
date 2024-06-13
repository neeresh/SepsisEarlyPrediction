import pandas as pd


def platelets_sofa(platelets):
    s_score = 0
    if platelets > 150:
        s_score += 0
    elif platelets >= 101 and platelets <= 150:
        s_score += 1
    elif platelets >= 51 and platelets <= 100:
        s_score += 2
    elif platelets >= 21 and platelets <= 50:
        s_score += 3
    elif platelets <= 20:
        s_score += 4

    return s_score


def total_bilirubin_sofa(bilirubin):
    s_score = 0
    if bilirubin < 1.2:
        s_score += 0
    elif bilirubin >= 1.2 and bilirubin <= 1.9:
        s_score += 1
    elif bilirubin >= 2.0 and bilirubin <= 5.9:
        s_score += 2
    elif bilirubin >= 6 and bilirubin <= 11.9:
        s_score += 3
    elif bilirubin >= 12.0:
        s_score += 4

    return s_score


def map_sofa(map):
    s_score = 0
    if map >= 70:
        s_score += 0
    elif map < 70:
        s_score += 1

    return s_score


def respiratory_rate_qsofa(respiratory_rate):
    q_score = 0
    if respiratory_rate >= 22.0:
        q_score += 1

    return q_score


def sbp_qsofa(sbp):
    q_score = 0
    if sbp < 100.0:
        q_score += 1

    return q_score


def q_sofa_indicator(row):
    resp = row['ResP_qSOFA']
    sbp = row['SBP_qSOFA']
    q_score = 0
    if resp > 0 and sbp > 0:
        q_score += 1
    return q_score


def sofa_indicator(row):
    # 2+ points indicates organ dysfunction
    platelets = row['Platelets_SOFA']
    bilirubin_total = row['Bilirubin_total_SOFA']
    map = row['MAP_SOFA']

    total_points = platelets + bilirubin_total + map

    q_score = 0
    if total_points > 2:
        q_score += 1
    return q_score


def mortality_sofa(row):
    # 2+ points indicates organ dysfunction
    platelets = row['Platelets_SOFA']
    bilirubin_total = row['Bilirubin_total_SOFA']
    map = row['MAP_SOFA']

    total_points = platelets + bilirubin_total + map

    mortality_rate = 0
    if total_points > 1 and total_points <= 9:
        mortality_rate += 0.30
    elif total_points >= 10 and total_points < 14:
        mortality_rate += 0.50
    elif total_points >= 14:
        mortality_rate += 0.95

    return mortality_rate


def temp_sirs(temp):
    sirs_score = 0
    if temp < 36 or temp >= 38:
        sirs_score += 1

    return sirs_score


def heart_rate_sirs(heart_rate):
    sirs_score = 0
    if heart_rate > 90:
        sirs_score += 1

    return sirs_score


def resp_sirs(resp):
    sirs_score = 0
    if resp > 20:
        sirs_score += 1

    return sirs_score


def paco2_sirs(paco2):
    sirs_score = 0
    if paco2 < 32:
        sirs_score += 1

    return sirs_score


def wbc_sirs(wbc):
    sirs_score = 0
    if wbc*1000 < 4000 or wbc*1000 > 12000:
        sirs_score += 1
    return sirs_score


if __name__ == '__main__':
    dataset = pd.read_pickle('../data/processed/training_ffill_bfill_zeros.pickle')
    for idx, patient_data in enumerate(dataset):
        if idx == 8:
            break

    patient_data['MAP_SOFA'] = patient_data['MAP'].apply(map_sofa)
    patient_data['Bilirubin_total_SOFA'] = patient_data['Bilirubin_total'].apply(total_bilirubin_sofa)
    patient_data['Platelets_SOFA'] = patient_data['Platelets'].apply(platelets_sofa)
    patient_data['ResP_qSOFA'] = patient_data['Resp'].apply(respiratory_rate_qsofa)
    patient_data['SBP_qSOFA'] = patient_data['SBP'].apply(sbp_qsofa)

    patient_data['qSOFA_indicator'] = patient_data.apply(q_sofa_indicator, axis=1)  # Sepsis detected
    patient_data['SOFA_indicator'] = patient_data.apply(sofa_indicator, axis=1)  # Organ Dysfunction occurred
    patient_data['Mortality_sofa'] = patient_data.apply(mortality_sofa, axis=1)  # Morality rate

    patient_data['Temp_sirs'] = patient_data['Temp'].apply(temp_sirs)
    patient_data['HR_sirs'] = patient_data['HR'].apply(heart_rate_sirs)
    patient_data['Resp_sirs'] = patient_data['Resp'].apply(resp_sirs)
    patient_data['paco2_sirs'] = patient_data['PaCO2'].apply(resp_sirs)
    patient_data['wbc_sirs'] = patient_data['WBC'].apply(wbc_sirs)

    print(f"Total number of features: {patient_data.shape[1]}:", patient_data.columns)


