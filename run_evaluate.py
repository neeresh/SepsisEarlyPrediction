import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from utils.evaluate_helper_methods import *
from utils.path_utils import project_root
from utils.evaluate_sepsis_score import evaluate_sepsis_score
from utils.get_true_labels import get_true_labels

from utils.helpers import get_features

import tqdm

d_input, d_channel, d_output = 336, 63, 2
scaler = MinMaxScaler()


def get_sepsis_score(data, model):

    columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH',
               'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
               'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct',
               'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime',
               'ICULOS']

    # Reformatting data into DataFrame to add features
    patient_data = pd.DataFrame(data, columns=columns)

    # Handling Missing values
    patient_data = patient_data.bfill()
    patient_data = patient_data.ffill()
    patient_data = patient_data.fillna(0)

    # Adding additional features (batch_size, time_steps, 40) -> (batch_size, time_steps, 63)
    # patient_data = add_additional_features_for_evaluation(patient_data)

    # Padding extra rows
    patient_data_length = [len(patient_data)]
    max_rows = 336
    num_features = patient_data.shape[1]
    if len(patient_data) < max_rows:
        padding = np.zeros((max_rows - len(patient_data), num_features))
        patient_data = np.vstack((patient_data.values, padding)).astype(np.float32)
    else:
        patient_data = patient_data.values.astype(np.float32)

    # Scaling
    patient_data = scaler.fit_transform(patient_data)
    patient_data = torch.from_numpy(patient_data).unsqueeze(0)

    # Predictions
    model.eval()
    model.to(device)
    predictions = []
    probas = []

    with torch.no_grad():
        patient_data = patient_data.to(torch.float32).to(device)
        outputs, _, _, _, _, _, _ = model(patient_data, patient_data_length, stage='test')

        _, predicted = torch.max(outputs, 1)
        probabilities = F.softmax(outputs, dim=1)

        predicted_class = predicted.detach().cpu().numpy()[0]

        predictions.append(predicted_class)
        probas.append(probabilities.detach().cpu().numpy()[0][predicted_class])

    return predictions, probas, patient_data


def evaluate():

    # Gathering Files
    input_directory = os.path.join(project_root(), 'physionet.org', 'files',
                                   'challenge-2019', '1.0.0', 'training', 'training_setA')
    output_directory = "./predictions_weighted_remove"

    # Find files.
    files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith(
                'psv'):
            files.append(f)

    # files.sort()
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # Load Sepsis Model
    model_path = "balanced_gtn_0.pkl"
    model = load_sepsis_model(d_input=d_input, d_channel=d_channel, d_output=d_output, model_name=model_path,
                              pre_model="modified_gtn")

    # Iterate over files.
    files = files[:3000]
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
            current_data = data[:t + 1]
            current_labels, current_score, data_df = get_sepsis_score(current_data, model)
            scores[t] = current_score[0]
            labels[t] = current_labels[0]

        output_file = os.path.join(output_directory, f)
        save_challenge_predictions(output_file, scores, labels)

    # get_true_labels(custom_files=files)


evaluate()

# Get true labels
# get_true_labels()

# Evaluate true and predicted labels

auroc, auprc, accuracy, f_measure, normalized_observed_utility = evaluate_sepsis_score(label_directory='./labels/',
                                                                                       prediction_directory='./predictions/')

print(f"Model's ability to distinguish between positive and negative classes (AUROC): {auroc}")
print(f"Model's precision-recall trade-off (AUPRC): {auprc}")
print(f"Model's overall accuracy: {accuracy}")
print(f"Model's balance between precision and recall (F-measure): {f_measure}")
print(f"Normalized utility score: {normalized_observed_utility}")

# logging.info(f"Model's ability to distinguish between positive and negative classes (AUROC): {auroc}")
# logging.info(f"Model's precision-recall trade-off (AUPRC): {auprc}")
# logging.info(f"Model's overall accuracy: {accuracy}")
# logging.info(f"Model's balance between precision and recall (F-measure): {f_measure}")
