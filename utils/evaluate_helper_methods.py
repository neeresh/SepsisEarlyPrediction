from utils.add_features import platelets_sofa, total_bilirubin_sofa, map_sofa, sofa_score, detect_sofa_change, \
    respiratory_rate_qsofa, sbp_qsofa, qsofa_score, q_sofa_indicator, sofa_indicator, detect_qsofa_change, \
    mortality_sofa, temp_sirs, heart_rate_sirs, resp_sirs, paco2_sirs, wbc_sirs, t_sofa, t_sepsis

from train_gtn import GatedTransformerNetwork, load_model, initialize_experiment
from utils.loader import make_loader

from utils.config import gtn_param
from torch.utils.data import DataLoader, ConcatDataset

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import tqdm

import numpy as np
import pandas as pd


device = 'cuda'


def load_sepsis_model(d_input, d_channel, d_output):
    """
    Used to load the trained model
    """
    config = gtn_param
    d_input, d_channel, d_output = d_input, d_channel, d_output  # (time_steps (window_size), channels, num_classes)
    model = GatedTransformerNetwork(d_model=config['d_model'], d_input=d_input, d_channel=d_channel,
                                    d_output=d_output, d_hidden=config['d_hidden'], q=config['q'],
                                    v=config['v'], h=config['h'], N=config['N'], dropout=config['dropout'],
                                    pe=config['pe'], mask=config['mask'], device=device).to(device)

    return load_model(model)


def load_challenge_data(file):
    """
    Parses .psv file and removes SepsisLabel (Target)
    returns: np.array
    """
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # Ignore SepsisLabel column if present.
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        data = data[:, :-1]

    return data


def save_challenge_predictions(file, scores, labels):
    """
    Writes output to the "predictions" directory
    Format: 
    PredictedProbability|PredictedLabel
        0.1|0
        0.2|0
        0.3|0
        0.8|1
        0.9|1
    """
    with open(file, 'w') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))


def t_suspicion(patient_data):
    """
    Since we don't have information about IV antibiotics and blood cultures,
    we are is considering that patient have infection if any 2 SIRS criteria are met

    Minor changes fron the original implementations
    """
    patient_data['PatientID'] = 0  # Just for groupby operation - created

    patient_data['infection_proxy'] = (
                patient_data[['Temp_sirs', 'HR_sirs', 'Resp_sirs']].eq(1).sum(axis=1) >= 2).astype(int)
    # t_suspicion is the first hour of (ICULOS) where infection proxy is positive at time t
    patient_data['t_suspicion'] = patient_data.groupby(['PatientID'])['ICULOS'].transform(
        lambda x: x[patient_data['infection_proxy'] == 1].min() if (patient_data['infection_proxy'] == 1).any() else 0)

    patient_data = patient_data.drop(['PatientID'], axis=1)  # Just for groupby operation - removed

    return patient_data
