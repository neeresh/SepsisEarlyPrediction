import logging
import numpy as np

import torch
import tqdm
from sklearn.model_selection import train_test_split
from torch import nn, optim

from models.modified_gtn import ModifiedGatedTransformerNetwork
from utils.config import gtn_param
from utils.loader import make_loader

import pandas as pd
import datetime

import os

import logging
from utils.path_utils import project_root


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

    # writer = SummaryWriter(log_dir=os.path.join(project_root(), 'data', 'logs', current_time), comment='')

    return training_examples, lengths_list, is_sepsis


def save_model(model, model_name):
    logging.info(f"Saving the model with model_name: {model_name}")

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    torch.save(model.state_dict(), model_name)

    logging.info(f"Saving successfull!!!")


def load_model(model, model_name="model_gtn.pkl"):
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


if __name__ == '__main__':

    print(f"Using {torch.cuda.device_count()} GPUs...")

    training_examples, lengths_list, is_sepsis = initialize_experiment('final_dataset.pickle')

    batch_size = 128 * torch.cuda.device_count()
    print(f"Batch size: {batch_size}")
    logging.info(f"Batch size: {batch_size}")

    sepsis = pd.Series(is_sepsis)
    positive_sepsis_idxs = sepsis[sepsis == 1].index
    negative_sepsis_idxs = sepsis[sepsis == 0].sample(frac=0.10, random_state=42).index
    all_samples = list(positive_sepsis_idxs) + list(negative_sepsis_idxs)
    np.random.shuffle(all_samples)

    print(f"Number of positive samples: {len(positive_sepsis_idxs)}")
    print(f"Number of negative samples: {len(negative_sepsis_idxs)}")

    print(f"Total samples: {len(all_samples)}")
    train_indicies, test_indicies = train_test_split(all_samples, test_size=0.20, random_state=42)
    train_loader, test_loader, train_indicies, test_indicies = make_loader(training_examples, lengths_list, is_sepsis,
                                                                           batch_size=batch_size, mode='padding_and_lengths',
                                                                           num_workers=8, train_indicies=train_indicies,
                                                                           test_indicies=test_indicies)

    criterion = nn.CrossEntropyLoss()

    device = 'cuda'
    config = gtn_param
    (d_input, d_channel), d_output = train_loader.dataset.data[0].shape, 2  # (time_steps, features, num_classes)
    print(f"d_input: {d_input}, d_channel: {d_channel}, d_output: {d_output}")

    print(d_input, d_channel, d_output)

    model = ModifiedGatedTransformerNetwork(d_model=config['d_model'], d_input=d_input, d_channel=d_channel,
                                            d_output=d_output, d_hidden=config['d_hidden'], q=config['q'],
                                            v=config['v'], h=config['h'], N=config['N'], dropout=config['dropout'],
                                            pe=config['pe'], mask=config['mask'], device=device).to(device)

    # model = nn.DataParallel(model)

    optimizer = optim.Adagrad(model.parameters(), lr=1e-4)  # GTN

    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []

    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        correct_train, total_train = 0, 0

        train_loader_tqdm = tqdm.tqdm(
            enumerate(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", total=len(train_loader))

        for idx, (inputs, labels, lengths) in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _, _, _, _, _, _ = model(inputs.to(torch.float32), lengths, 'train')

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Update tqdm description for training progress
            train_loader_tqdm.set_postfix({
                "Train Loss": running_train_loss / total_train,
                "Train Acc": correct_train / total_train
            })

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_accuracy = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        # Testing phase
        model.eval()
        running_test_loss = 0.0
        correct_test, total_test = 0, 0

        with torch.no_grad():
            for idx, (inputs, labels, lengths) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _, _, _, _, _, _ = model(inputs.to(torch.float32), lengths, 'test')
                loss = criterion(outputs, labels)
                running_test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        epoch_test_accuracy = correct_test / total_test
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_accuracy)

        message = f"Epoch {epoch + 1}/{epochs} - " \
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}, " \
                  f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_accuracy:.4f}"

        tqdm.tqdm.write(message)

        save_model(model, model_name=f"./saved_models/modified_gtn/modified_gtn_{epoch}.pkl")

    save_model(model, model_name=f"./saved_models/modified_gtn/modified_gtn.pkl")
