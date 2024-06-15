import numpy as np
import pandas as pd

from utils.evaluate_helper_methods import *
from utils.path_utils import project_root

import tqdm


d_input, d_channel, d_output = 336, 63, 2

def get_sepsis_score(data, model):

    columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp',
               'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST',
               'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
               'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
               'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
               'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
               'HospAdmTime', 'ICULOS']

    patient_data = pd.DataFrame(data, columns=columns).fillna(0)

    # Vectorized feature calculations
    patient_data['MAP_SOFA'] = map_sofa(patient_data['MAP'])
    patient_data['Bilirubin_total_SOFA'] = total_bilirubin_sofa(patient_data['Bilirubin_total'])
    patient_data['Platelets_SOFA'] = platelets_sofa(patient_data['Platelets'])
    patient_data['SOFA_score'] = patient_data[['MAP_SOFA', 'Bilirubin_total_SOFA', 'Platelets_SOFA']].sum(axis=1)
    patient_data = detect_sofa_change(patient_data)

    patient_data['ResP_qSOFA'] = respiratory_rate_qsofa(patient_data['Resp'])
    patient_data['SBP_qSOFA'] = sbp_qsofa(patient_data['SBP'])
    patient_data['qSOFA_score'] = patient_data[['ResP_qSOFA', 'SBP_qSOFA']].sum(axis=1)

    # Assuming the q_sofa_indicator, sofa_indicator, and mortality_sofa are predefined
    patient_data['qSOFA_indicator'] = patient_data.apply(q_sofa_indicator, axis=1)
    patient_data['SOFA_indicator'] = patient_data.apply(sofa_indicator, axis=1)
    patient_data['Mortality_sofa'] = patient_data.apply(mortality_sofa, axis=1)
    
    # Padding remaning rows to meet the model requirements
    # Each patient file will be (336, 63) -> (Timestamps, features)

    # 336 rows are padded dynamically based on how each timestamp for each patient
    max_rows = 336
    num_features = patient_data.shape[1]
    if len(patient_data) < max_rows:
        padding = np.zeros((max_rows - len(patient_data), num_features))
        patient_data = np.vstack((patient_data.values, padding))
        
    elif len(patient_data) > max_rows:
        patient_data = patient_data.iloc[:max_rows].values
    else:
        patient_data = patient_data.values


    patient_data = torch.tensor(patient_data).unsqueeze(0)
    
    model.eval()
    model.to(device)
    predictions = []
    probas = []

    with torch.no_grad():
        patient_data = patient_data.to(torch.float32).to(device)
        outputs, _, _, _, _, _, _ = model(patient_data, stage='test')
    
        _, predicted = torch.max(outputs, 1)
        probabilities = F.softmax(outputs, dim=1)
        
        predicted_class = predicted.detach().cpu().numpy()[0]

        predictions.append(predicted_class)
        probas.append(probabilities.detach().cpu().numpy()[0][predicted_class])

    return predictions, probas, patient_data

def evaluate():

    # Gathering Files
    # input_directory = os.path.join(project_root(), 'physionet.org', 'files', 'challenge-2019', '1.0.0', 'training','training_setA')
    
    input_directory = "/localscratch/neeresh/data/physionet2019/physionet.org/files/challenge-2019/1.0.0/training/training_setA/"
    # input_directory = "/localscratch/neeresh/data/physionet2019/physionet.org/files/challenge-2019/1.0.0/training/training_setB/"
    output_directory = "./predictions"

    # Find files.
    files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
            files.append(f)
    
    # files.sort()
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    
    # Load Sepsis Model
    model = load_sepsis_model(d_input=d_input, d_channel=d_channel, d_output=d_output)

    # Iterate over files.
    print('Predicting sepsis labels...')
    num_files = len(files)
    for i, f in tqdm.tqdm(enumerate(files), desc="Remaining Files: ", total=num_files):
        # print('    {}/{}...'.format(i+1, num_files))

        # Load data.
        input_file = os.path.join(input_directory, f)
        data = load_challenge_data(input_file)

        # Make predictions.
        num_rows = len(data)  # Number of patient recordings
        scores = np.zeros(num_rows)
        labels = np.zeros(num_rows)
        
        for t in range(num_rows):
            current_data = data[:t+1]
            current_labels, current_score, data_df = get_sepsis_score(current_data, model)
            scores[t] = current_score[0]
            labels[t] = current_labels[0]
        
        output_file = os.path.join(output_directory, f)
        save_challenge_predictions(output_file, scores, labels)
    
    return model, data, current_data, data_df

model, data, current_data, data_df = evaluate()

from utils.evaluate_sepsis_score import evaluate_sepsis_score

# Numbers of label and prediction files must be the same
evaluate_sepsis_score(label_directory='./labels/', prediction_directory='./predictions/')

