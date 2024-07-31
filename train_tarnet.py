import datetime
import logging
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter

from utils.config import tarnet_param
from utils.loader import make_loader
from utils.path_utils import project_root

from models.tarnet.utils import initialize_training
from models.tarnet.utils import training, preprocess

config = tarnet_param

from models.tarnet.multitask_transformer_class import MultitaskTransformerModel


def _setup_destination(current_time):
    log_path = os.path.join(project_root(), 'data', 'logs', current_time)
    os.mkdir(log_path)
    logging.basicConfig(filename=os.path.join(log_path, current_time + '.log'), level=logging.DEBUG)

    return log_path


def _log(message: str = '{}', value: any = None):
    print(message.format(value))
    logging.info(message.format(value))


def initialize_experiment(data_file):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    destination_path = _setup_destination(current_time)

    _log(message="Datafile used: {}".format(data_file))

    # [[patient1], [patient2], [patient3], ..., [patientN]]
    training_examples = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', data_file))
    _log(message="Total number of patients: {}", value=len(training_examples))

    with open(os.path.join(project_root(), 'data', 'processed', 'lengths.txt')) as f:
        lengths_list = [int(length) for length in f.read().splitlines()]
    _log(message="Min recordings: {} & Max recordings: {}".format(min(lengths_list), max(lengths_list)))

    with open(os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt')) as f:
        is_sepsis = [int(is_sep) for is_sep in f.read().splitlines()]
    _log(message="Distribution of the SepsisLabel: \n{}".format(pd.Series(is_sepsis).value_counts()))

    writer = SummaryWriter(log_dir=os.path.join(project_root(), 'data', 'logs', current_time), comment='')

    return training_examples, lengths_list, is_sepsis, writer, destination_path


def save_model(model, model_name):
    logging.info(f"Saving the model with model_name: {model_name}")

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    torch.save(model.state_dict(), model_name)

    logging.info(f"Saving successfull!!!")


def load_model(model, model_name):
    device = 'cuda'
    print(f"Loading {model_name} GTN model...")
    logging.info(f"Loading GTN model...")
    model.load_state_dict(torch.load(model_name))

    print(f"Model is set to eval() mode...")
    logging.info(f"Model is set to eval() mode...")
    model.eval()

    print(f"Model is on the deivce: {device}")
    logging.info(f"Model is on the deivce: {device}")
    model.to(device)

    return model


def get_data_from_loaders(loader):

    X_list, y_list = [], []
    for batch in loader:
        X, y = batch
        X_list.append(X)
        y_list.append(y)

    # return torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)
    return np.concatenate(X_list), np.concatenate(y_list)


if __name__ == '__main__':
    print(f"Using {torch.cuda.device_count()} GPUs...")
    logging.info(f"Using {torch.cuda.device_count()} GPUs...")

    # Getting Data and Loaders
    data_file = "final_dataset.pickle"
    training_examples, lengths_list, is_sepsis, writer, destination_path = initialize_experiment(data_file)

    sepsis = pd.Series(is_sepsis)
    positive_sepsis_idxs = sepsis[sepsis == 1].index
    negative_sepsis_idxs = sepsis[sepsis == 0].sample(frac=0.20).index
    all_samples = list(positive_sepsis_idxs) + list(negative_sepsis_idxs)
    np.random.shuffle(all_samples)

    logging.info(f"Number of positive samples: {len(positive_sepsis_idxs)}")
    logging.info(f"Number of negative samples: {len(negative_sepsis_idxs)}")

    # Reducing the samples to have balanced dataset
    batch_size = config['batch'] * torch.cuda.device_count()
    print(f"Batch size: {batch_size}")
    logging.info(f"Batch size: {batch_size}")

    # Splitting dataset into train and test
    logging.info(f"Total samples: {len(all_samples)}")

    train_indicies, temp_indicies = train_test_split(all_samples, test_size=0.2, random_state=42)  # 80 20
    val_indicies, test_indicies = train_test_split(temp_indicies, test_size=0.5, random_state=42)  # 10 10

    train_loader, val_loader, test_loader, train_indicies, val_indices, test_indicies = make_loader(
        training_examples, lengths_list, is_sepsis, batch_size=batch_size, mode='padding', num_workers=4,
        train_indicies=train_indicies, test_indicies=test_indicies, val_indicies=val_indicies,
        select_important_features=False, include_val=True)

    logging.info(f"Number of training samples: {len(train_indicies)}")
    logging.info(f"Number of validation samples: {len(val_indicies)}")
    logging.info(f"Number of test samples: {len(test_indicies)}")

    # Model's input shape
    (d_input, d_channel), d_output = train_loader.dataset.data[0].shape, 2  # (time_steps, features, num_classes)
    logging.info(f"d_input: {d_input}, d_channel: {d_channel}, d_output: {d_output}")
    num_epochs = config['epochs']

    print(d_input, d_channel, d_output)

    logging.info(config)
    logging.info(f"d_input: {d_input}, d_channel: {d_channel}, d_output: {d_output}")
    logging.info(f"Number of epochs: {num_epochs}")

    # (d_input, d_channel), d_output = (336, 191), 2
    model = MultitaskTransformerModel(task_type=config['task_type'], device=config['device'],
                                      nclasses=d_output, seq_len=d_input, batch=config['batch'],
                                      input_size=d_channel, emb_size=config['emb_size'],
                                      nhead=config['nhead'], nhid=config['nhid'], nhid_tar=config['nhid_tar'],
                                      nhid_task=config['nhid_task'], nlayers=config['nlayers'],
                                      dropout=config['dropout'], )

    logging.info(model)

    model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer = initialize_training(tarnet_param)

    X_train, y_train = get_data_from_loaders(train_loader)  # torch.Size([8330, 336, 191]) torch.Size([8330])
    X_val, y_val = get_data_from_loaders(val_loader)  # torch.Size([1041, 336, 191]) torch.Size([1041])
    X_test, y_test = get_data_from_loaders(test_loader)  # torch.Size([1042, 336, 191]) torch.Size([1042])

    # Pre-processing and making perfect batches; here we pad zeros to have equal samples in the last batch
    X_train, y_train, X_test, y_test = preprocess(tarnet_param, X_train, y_train, X_test, y_test)

    logging.info(f"X_train: {type(X_train)}, y_train: {type(y_train)}")
    logging.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")

    logging.info('Training start...')
    model = training(model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer, X_train,
                   y_train, X_test, y_test, tarnet_param)
    logging.info('Training complete...')

    save_model(model, model_name=f"./saved_models/tarnet/tarnet_final_{config['epochs']}_val.pkl")
