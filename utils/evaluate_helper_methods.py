import os.path

from torch import nn

# from models.simmtm.config import Config
from models.simmtm.model import TFC, target_classifier
from models.tarnet.multitask_transformer_class import MultitaskTransformerModel

from training_scripts.train_gtn import load_model
from models.custom_models.modified_gtn import ModifiedGatedTransformerNetwork  # Modified GTN
from models.custom_models.gtn_mask import MaskedGatedTransformerNetwork
from models.gtn.transformer import Transformer  # Original GTN
from utils.helpers import get_features

from utils.config import gtn_param, tarnet_param, masked_gtn_param, pretrain_params, modified_gtn_param

import torch

import pandas as pd

from utils.path_utils import project_root
from utils.pretrain_utils.get_args import get_args

device = 'cuda'

from models.adatime.da.models import get_backbone_class, GTN, classifier, codats_classifier
from models.adatime.da.algorithms import get_algorithm_class

import os
from utils.path_utils import project_root
import torch


def load_checkpoint_da(da_name, ckp_type, config):
    model_path = os.path.join(project_root(), 'results', 'adatime', f'{da_name}',
                              'pretrain_finetune', f'{da_name}_{da_name}_gtn', '0_to_1_run_0',
                              'checkpoint.pt')

    print(f"Loading model from {model_path}...")
    pretrained_dict = torch.load(model_path)

    if ckp_type == 'last':
        print(f"Loading 'last' checkpoint from {model_path}...")
        pretrained_model = pretrained_dict['last']
    else:
        print(f"Loading 'best' checkpoint from {model_path}...")
        pretrained_model = pretrained_dict['best']

    # Original Model
    original_feature_extactor = GTN(config)

    if da_name == 'CoDATS':
        print(f"Loading CoDATS classifier...")
        original_classifier = codats_classifier(config)
    else:
        original_classifier = classifier(config)

    # pre-trained model
    model = nn.Sequential(original_feature_extactor, original_classifier)
    model.load_state_dict(pretrained_model)  # Loading weights

    return model


# def preprocessing(patient_data):
#     columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH',
#                'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
#                'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct',
#                'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime',
#                'ICULOS']
#
#     patient_data = pd.DataFrame(patient_data, columns=columns)
#     patient_data = add_feature_informative_missingness(patient_data)
#     patient_data = fill_missing_values(patient_data)
#     patient_data = add_sliding_features_for_vital_signs(patient_data)
#     patient_data = add_additional_features(patient_data)
#
#     return patient_data


def load_model(model, model_name="model_gtn.pkl"):
    device = 'cuda'

    print(f"Loading {model_name}...")
    model.load_state_dict(torch.load(model_name))

    print(f"Model is set to eval() mode...")
    model.eval()

    print(f"Model is on the deivce: {device}")
    model.to(device)

    return model


def load_sepsis_model(d_input, d_channel, d_output, model_name, pre_model, da_ckp_type='best'):
    """
    Used to load the trained model
    """
    if pre_model == 'gtn':
        config = gtn_param
        print(f"Loading from {model_name}...")
        print(f"Loading original model...")

        model = Transformer(d_model=config['d_model'], d_input=d_input, d_channel=d_channel,
                            d_output=d_output, d_hidden=config['d_hidden'], q=config['q'],
                            v=config['v'], h=config['h'], N=config['N'], dropout=config['dropout'],
                            pe=config['pe'], mask=config['mask'], device=device).to(device)

        return load_model(model, model_name)

    elif pre_model == 'masked_gtn':
        config = masked_gtn_param
        print(f"Loading from {model_name}...")
        print("Loading modified gtn model")
        model = MaskedGatedTransformerNetwork(d_model=config['d_model'], d_input=d_input, d_channel=d_channel,
                                              d_output=d_output, d_hidden=config['d_hidden'], q=config['q'],
                                              v=config['v'], h=config['h'], N=config['N'], dropout=config['dropout'],
                                              pe=config['pe'], mask=config['mask'], device=device).to(device)

        return load_model(model, model_name)

    elif pre_model == 'modified_gtn':
        config = modified_gtn_param
        print(f"Loading from {model_name}...")
        print("Loading modified gtn model")
        model = ModifiedGatedTransformerNetwork(d_model=config['d_model'], d_input=d_input, d_channel=d_channel,
                                                d_output=d_output, d_hidden=config['d_hidden'], q=config['q'],
                                                v=config['v'], h=config['h'], N=config['N'], dropout=config['dropout'],
                                                pe=config['pe'], mask=config['mask'], device=device).to(device)

        return load_model(model, model_name)

    elif pre_model == 'tarnet':
        print(f"Loading from {model_name}...")
        print("Loading tarnet model")

        config = tarnet_param
        # (d_input, d_channel), d_output = (336, 191), 2

        model = MultitaskTransformerModel(task_type=config['task_type'], device=config['device'],
                                          nclasses=d_output, seq_len=d_input, batch=16,
                                          input_size=d_channel, emb_size=config['emb_size'],
                                          nhead=config['nhead'], nhid=config['nhid'], nhid_tar=config['nhid_tar'],
                                          nhid_task=config['nhid_task'], nlayers=config['nlayers'],
                                          dropout=config['dropout'], )

        return load_model(model, model_name)

    elif pre_model == 'pretrained_gtn':

        from models.gtn.config import Config
        from models.gtn_modified.transformer import Transformer

        config = Config()
        # args, unknown = get_args()

        chkpoint = torch.load(model_name)  # Model Path
        print(f"Loading pre-trained model: {pre_model} from {model_name}...")

        model = Transformer(d_model=config.d_model, d_input=config.d_input,
                            d_channel=config.d_channel, d_output=config.d_output,
                            d_hidden=config.d_hidden, q=config.q, v=config.v,
                            h=config.h, N=config.N, device=config.device,
                            dropout=config.dropout, pe=config.pe, mask=config.mask)

        pretrained_dict = chkpoint["model_state_dict"]
        # model_dict = model.state_dict()
        # model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)

        return model

        # config = pretrain_params
        # print(f"Loading from {model_name}...")
        # print(f"Loading original model...")
        #
        # model = Transformer(d_model=config['d_model'], d_input=d_input, d_channel=d_channel,
        #                     d_output=d_output, d_hidden=config['d_hidden'], q=config['q'],
        #                     v=config['v'], h=config['h'], N=config['N'], dropout=config['dropout'],
        #                     pe=config['pe'], mask=config['mask'], device=device).to(device)
        #
        # return load_model(model, model_name)

    elif pre_model == 'simmtm':

        from models.simmtm.config import Config

        config = Config()
        args, unknown = get_args()

        chkpoint = torch.load(model_name)  # Model Path
        print(f"Loading pre-trained model: {pre_model} from {model_name}...")

        model = TFC(configs=config, args=args)
        pretrained_dict = chkpoint["model_state_dict"]
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        classifier = target_classifier(configs=config)
        pretrained_classifier_dict = chkpoint["classifier"]
        classifier_dict = classifier.state_dict()
        classifier_dict.update(pretrained_classifier_dict)
        classifier.load_state_dict(classifier_dict)

        return model, classifier

    elif pre_model == 'pretrained_tfc':

        from models.tfc.config import Config
        from models.tfc.model import TFC, target_classifier

        config = Config()
        args, unknown = get_args()

        chkpoint = torch.load(model_name)  # Model Path
        print(f"Loading pre-trained model: {pre_model} from {model_name}...")

        model = TFC(configs=config)
        pretrained_dict = chkpoint["model_state_dict"]
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print(f"Loading pre-trained classifier: {pre_model} from {model_name}...")
        classifier = target_classifier(configs=config)
        pretrained_classifier_dict = chkpoint["classifier"]
        classifier_dict = classifier.state_dict()
        classifier_dict.update(pretrained_classifier_dict)
        classifier.load_state_dict(classifier_dict)

        return model, classifier

    elif pre_model == 'da':

        from models.adatime.configs.get_configs import Config
        config = Config()

        pretrained_model = load_checkpoint_da(da_name=model_name, ckp_type=da_ckp_type, config=config)

        return pretrained_model

    else:
        ValueError(f"Couldn't find requested model: {model_name}")


# def load_challenge_data(file):
#     """
#     Parses .psv file and removes SepsisLabel (Target)
#     returns: np.array
#     """
#
#     with open(file, 'r') as f:
#         header = f.readline().strip()
#         column_names = header.split('|')
#         data = np.loadtxt(f, delimiter='|')
#
#     # Ignore SepsisLabel column if present.
#     if column_names[-1] == 'SepsisLabel':
#         column_names = column_names[:-1]
#         data = data[:, :-1]
#
#     return data

def load_challenge_data(file):
    """
    Parses .psv file and removes SepsisLabel (Target)
    returns: np.array
    """
    # Use pandas to read the file
    df = pd.read_csv(file, sep='|')

    # Check if the last column is 'SepsisLabel' and drop it if present
    if 'SepsisLabel' in df.columns:
        df = df.drop(columns=['SepsisLabel'])

    # Convert the DataFrame to a numpy array
    data = df.to_numpy()

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

    Minor changes from the original implementations
    """
    patient_data['PatientID'] = 0  # Just for groupby operation - created

    patient_data['infection_proxy'] = (
            patient_data[['Temp_sirs', 'HR_sirs', 'Resp_sirs']].eq(1).sum(axis=1) >= 2).astype(int)
    # t_suspicion is the first hour of (ICULOS) where infection proxy is positive at time t
    patient_data['t_suspicion'] = patient_data.groupby(['PatientID'])['ICULOS'].transform(
        lambda x: x[patient_data['infection_proxy'] == 1].min() if (patient_data['infection_proxy'] == 1).any() else 0)

    patient_data = patient_data.drop(['PatientID'], axis=1)  # Just for groupby operation - removed

    return patient_data


# def add_additional_features_for_evaluation(patient_data):
#     patient_data['MAP_SOFA'] = map_sofa(patient_data['MAP'])
#     patient_data['Bilirubin_total_SOFA'] = patient_data['Bilirubin_total'].apply(total_bilirubin_sofa)
#     patient_data['Platelets_SOFA'] = patient_data['Platelets'].apply(platelets_sofa)
#     patient_data['SOFA_score'] = patient_data.apply(sofa_score, axis=1)
#     patient_data = detect_sofa_change(patient_data)
#
#     patient_data['ResP_qSOFA'] = patient_data['Resp'].apply(respiratory_rate_qsofa)
#     patient_data['SBP_qSOFA'] = patient_data['SBP'].apply(sbp_qsofa)
#     patient_data['qSOFA_score'] = patient_data.apply(qsofa_score, axis=1)
#     patient_data = detect_qsofa_change(patient_data)
#
#     patient_data['qSOFA_indicator'] = patient_data.apply(q_sofa_indicator, axis=1)  # Sepsis detected
#     patient_data['SOFA_indicator'] = patient_data.apply(sofa_indicator, axis=1)  # Organ Dysfunction occurred
#     patient_data['Mortality_sofa'] = patient_data.apply(mortality_sofa, axis=1)  # Morality rate
#
#     patient_data['Temp_sirs'] = patient_data['Temp'].apply(temp_sirs)
#     patient_data['HR_sirs'] = patient_data['HR'].apply(heart_rate_sirs)
#     patient_data['Resp_sirs'] = patient_data['Resp'].apply(resp_sirs)
#     patient_data['paco2_sirs'] = patient_data['PaCO2'].apply(resp_sirs)
#     patient_data['wbc_sirs'] = patient_data['WBC'].apply(wbc_sirs)
#
#     patient_data = t_suspicion(patient_data)
#     patient_data = t_sofa(patient_data)
#     patient_data['t_sepsis'] = patient_data.apply(t_sepsis, axis=1)
#
#     # # NEWS - National Early Warning Score
#     patient_data['HR_NEWS'] = hr_news(patient_data['HR'])
#     patient_data['Temp_NEWS'] = temp_news(patient_data['Temp'])
#     patient_data['Resp_NEWS'] = resp_news(patient_data['Resp'])
#     patient_data['Creatinine_NEWS'] = creatinine_news(patient_data['Creatinine'])
#     patient_data['MAP_NEWS'] = map_news(patient_data['MAP'])
#
#     return patient_data


def remove_unwanted_features_for_evaluation(patient_data):  # Check and remove
    additional_features = ['MAP_SOFA', 'Bilirubin_total_SOFA', 'Platelets_SOFA', 'SOFA_score', 'SOFA_score_diff',
                           'SOFA_deterioration', 'ResP_qSOFA', 'SBP_qSOFA', 'qSOFA_score', 'qSOFA_score_diff',
                           'qSOFA_deterioration', 'qSOFA_indicator', 'SOFA_indicator', 'Mortality_sofa',
                           'Temp_sirs', 'HR_sirs', 'Resp_sirs', 'paco2_sirs', 'wbc_sirs']

    vital_signs, laboratory_values, demographics = get_features(case=1)
    final_features = vital_signs + laboratory_values + demographics + additional_features

    patient_data = patient_data[final_features]

    return patient_data
