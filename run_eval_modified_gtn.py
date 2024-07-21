import os

import numpy as np
import pandas as pd
import torch

from utils.evaluate_helper_methods import save_challenge_predictions, load_challenge_data, load_sepsis_model
from utils.path_utils import project_root
from utils.evaluate_sepsis_score import evaluate_sepsis_score

import torch.nn.functional as F
from utils.preprocessing import preprocessing, pad_rows

import tqdm

device = 'cuda'
d_input, d_channel, d_output = 336, 191, 2

def prepare_test_data():

    # Gather sepsis details
    file_path = os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt')
    sepsis = pd.Series(open(file_path, 'r').read().splitlines()).astype(int)

    # Collect all samples
    csv_path = os.path.join(project_root(), 'data', 'csv')
    training_files = [os.path.join(csv_path, f) for f in os.listdir(csv_path) if f.endswith('.csv')]
    training_files.sort()

    # Filtering only positive samples
    positive_sepsis = sepsis[sepsis == 1].index
    # negative_sepsis = sepsis[sepsis == 0].index

    # Filtered positive patients
    training_files = [training_files[idx] for idx in positive_sepsis]

    # File names
    file_names = [training_files[idx].split('/')[-1].replace('.csv', '.psv') for idx in range(len(training_files))]
    file_names.sort()

    # Converting them to psv files
    destination_path = os.path.join(project_root(), 'data', 'test_data')
    for i, (patient_data, file_name) in enumerate(
            tqdm.tqdm(zip(training_files, file_names), desc="Processing Positive Sepsis Patients",
                      total=len(training_files))):
        patient_data = pd.read_csv(patient_data)
        patient_data = patient_data.drop(['PatientID'], axis=1)
        patient_data.to_csv(os.path.join(destination_path, file_name), sep='|', index=False)

    # True labels - (For evaluation)
    destination_path = os.path.join(os.getcwd(), 'labels_modified_gtn')
    for i, (patient_data, file_name) in enumerate(
            tqdm.tqdm(zip(training_files, file_names), desc="Processing True Predictions",
                      total=len(training_files))):
        patient_data = pd.read_csv(patient_data)
        patient_data = patient_data['SepsisLabel']
        patient_data.to_csv(os.path.join(destination_path, file_name), sep='|', index=False)

    print(f"Test data is all set!!!.\nNumber of patients: {len(training_files)}")


def get_sepsis_score(data, model):

    columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH',
               'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
               'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct',
               'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime',
               'ICULOS']

    patient_data = pd.DataFrame(data, columns=columns)

    # length parameter for modified gtn
    seq_length = [len(data)]

    # Given input is preprocessed
    patient_data = preprocessing(patient_data=patient_data)

    # Padding extra rows
    patient_data = pad_rows(patient_data=patient_data)
    patient_data = torch.from_numpy(patient_data)  # (336, 191)
    patient_data = patient_data.unsqueeze(0)

    # Predictions
    model.eval()
    model.to(device)
    predictions = []
    probas = []

    with torch.no_grad():
        patient_data = patient_data.to(device)
        outputs, _, _, _, _, _, _ = model(patient_data, seq_length, stage='test')

        _, predicted = torch.max(outputs, 1)
        probabilities = F.softmax(outputs, dim=1)

        predicted_class = predicted.detach().cpu().numpy()[0]

        predictions.append(predicted_class)
        probas.append(probabilities.detach().cpu().numpy()[0][predicted_class])

    return predictions, probas, patient_data


def evaluate():

    # Gathering Files
    # Test data and true labels are created (only positive samples)
    prepare_test_data()

    input_directory = os.path.join(project_root(), 'data', 'test_data')
    output_directory = "./predictions_modified_gtn/"

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
    # (Trained on entire sequences (normal gtn) but predicting using (modified gtn))
    model_path = "./saved_models/gtn/gtn_final.pkl"
    model = load_sepsis_model(d_input=d_input, d_channel=d_channel, d_output=d_output, model_name=model_path,
                              pre_model="modified_gtn")

    # Iterate over files.
    # files = files[:3000]
    print('Predicting sepsis labels...')
    num_files = len(files)
    print(f"Total number of files: {num_files}")
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

evaluate()

# Evaluate true and predicted labels
auroc, auprc, accuracy, f_measure, normalized_observed_utility = evaluate_sepsis_score(
    label_directory='./labels_modified_gtn/', prediction_directory='./predictions_modified_gtn/')

print(f"Model's ability to distinguish between positive and negative classes (AUROC): {auroc}")
print(f"Model's precision-recall trade-off (AUPRC): {auprc}")
print(f"Model's overall accuracy: {accuracy}")
print(f"Model's balance between precision and recall (F-measure): {f_measure}")
print(f"Normalized utility score: {normalized_observed_utility}")
