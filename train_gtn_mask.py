from typing import Optional

import logging
import numpy as np

import torch
import tqdm
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader

from models.custom_models.gtn_mask import MaskedGatedTransformerNetwork
from utils.config import masked_gtn_param
from utils.loader import make_loader

import pandas as pd
import datetime

import os

import logging
from utils.path_utils import project_root
from utils.plot_metrics import plot_losses_and_accuracies

device = 'cuda'
config = masked_gtn_param


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


def train_model(model, train_loader: DataLoader, test_loader: DataLoader, epochs: int, class_0_weight=None,
                class_1_weight=None, val_loader: Optional[DataLoader] = None):

    # Use different class weights if specified
    if class_0_weight is not None and class_1_weight is not None:
        print(f"Using manual weights for classes 0 and 1")
        logging.info(f"Class0 weight: {class_0_weight} & Class1 weight: {class_1_weight}")
        manual_weights = torch.tensor([class_0_weight, class_1_weight]).to(device)

        label_smoothing = 0.05
        print(f"Using label smoothing of {label_smoothing}")
        logging.info(f"Using label smoothing of {label_smoothing}")
        criterion = nn.CrossEntropyLoss(weight=manual_weights, label_smoothing=label_smoothing)

    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adagrad(model.parameters(), lr=config['lr'])  # GTN
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)  # Not in GTN Implementation

    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        correct_train, total_train = 0, 0

        train_loader_tqdm = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for idx, (inputs, labels, padded_masks) in enumerate(train_loader_tqdm):
            optimizer.zero_grad()

            outputs, _, _, _, _, _, _ = model(inputs.to(device).to(torch.float32), 'train', padded_masks)
            loss = criterion(outputs, labels.to(device))

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted.detach().cpu() == labels).sum().item()

            # Update tqdm description for training progress
            train_loader_tqdm.set_postfix({
                "Train Loss": running_train_loss / total_train,
                "Train Acc": correct_train / total_train
            })

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_accuracy = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        # Validation phase
        if val_loader:
            model.eval()
            running_val_loss = 0.0
            correct_val, total_val = 0, 0

            with torch.no_grad():
                for idx, (inputs, labels, padded_masks) in enumerate(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs, _, _, _, _, _, _ = model(inputs.to(device).to(torch.float32), 'test', padded_masks)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted.detach().cpu() == labels).sum().item()

            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            epoch_val_accuracy = correct_val / total_val
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_accuracy)
        else:
            epoch_val_loss = "N/A"
            epoch_val_accuracy = "N/A"

        # epoch_val_loss = "N/A"  # Remove when uncommenting above code
        # epoch_val_accuracy = "N/A"  # Remove when uncommenting above code

        # Testing phase
        running_test_loss = 0.0
        correct_test, total_test = 0, 0

        with torch.no_grad():
            model.eval()
            for idx, (inputs, labels, padded_masks) in enumerate(test_loader):
                outputs, _, _, _, _, _, _ = model(inputs.to(device).to(torch.float32), 'test', padded_masks)
                loss = criterion(outputs, labels.to(device))
                running_test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted.detach().cpu() == labels).sum().item()

        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        epoch_test_accuracy = correct_test / total_test
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_accuracy)

        # scheduler.step()  # Not in GTN Implementation

        message = f"Epoch {epoch + 1}/{epochs} - " \
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}, " \
                  f"Val Loss: {epoch_val_loss}, Val Acc: {epoch_val_accuracy}, " \
                  f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_accuracy:.4f}"
        tqdm.tqdm.write(message)
        logging.info(message)

    # Saving the model
    save_model(model, model_name=f"./saved_models/masked_gtn/masked_gtn_final_{config['num_epochs']}_val.pkl")

    return {"train_loss": train_losses, "val_loss": val_losses if val_loader else None, "test_loss": test_losses,
            "train_accuracy": train_accuracies, "val_accuracy": val_accuracies if val_loader else None,
            "test_accuracy": test_accuracies}


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

    print(f"Number of positive samples: {len(positive_sepsis_idxs)}")
    print(f"Number of negative samples: {len(negative_sepsis_idxs)}")

    # Reducing the samples to have balanced dataset
    batch_size = config['batch_size'] * torch.cuda.device_count()
    print(f"Batch size: {batch_size}")
    logging.info(f"Batch size: {batch_size}")

    # Splitting dataset into train and test
    print(f"Total samples: {len(all_samples)}")

    train_indicies, temp_indicies = train_test_split(all_samples, test_size=0.2, random_state=42)  # 80 20
    val_indicies, test_indicies = train_test_split(temp_indicies, test_size=0.5, random_state=42)  # 10 10

    train_loader, val_loader, test_loader, train_indicies, val_indices, test_indicies = make_loader(
        training_examples, lengths_list, is_sepsis, batch_size=batch_size, mode='padding_masking', num_workers=4,
        train_indicies=train_indicies, test_indicies=test_indicies, val_indicies=val_indicies,
        select_important_features=False, include_val=True)

    # Model's input shape
    (d_input, d_channel), d_output = train_loader.dataset.data[0].shape, 2  # (time_steps, features, num_classes)
    print(f"d_input: {d_input}, d_channel: {d_channel}, d_output: {d_output}")
    num_epochs = config['num_epochs']

    print(d_input, d_channel, d_output)

    logging.info(config)
    logging.info(f"d_input: {d_input}, d_channel: {d_channel}, d_output: {d_output}")
    logging.info(f"Number of epochs: {num_epochs}")

    model = MaskedGatedTransformerNetwork(d_model=config['d_model'], d_input=d_input, d_channel=d_channel,
                                          d_output=d_output, d_hidden=config['d_hidden'], q=config['q'],
                                          v=config['v'], h=config['h'], N=config['N'], dropout=config['dropout'],
                                          pe=config['pe'], mask=config['mask'], device=device).to(device)

    model = nn.DataParallel(model)

    # Training the model and saving metrics
    metrics = train_model(model, train_loader, val_loader, epochs=num_epochs)

    # Save test files in /localhost/.../test_data
    import glob

    test_data_path = os.path.join(project_root(), 'data', 'test_data', 'masked_gtn')
    os.makedirs(test_data_path, exist_ok=True)

    # Removing all files before uploading test data
    for file_path in glob.glob(f"{test_data_path}/*"):
        os.remove(file_path)

    # Gathering all files
    files_names = []
    input_directory = [
        os.path.join(project_root(), 'physionet.org', 'files', 'challenge-2019', '1.0.0', 'training', 'training_setA'),
        os.path.join(project_root(), 'physionet.org', 'files', 'challenge-2019', '1.0.0', 'training', 'training_setB')
    ]

    # Loading all file names and sort
    for dir in input_directory:
        for f in os.listdir(dir):
            file_path = os.path.join(dir, f)
            if os.path.isfile(file_path) and not f.lower().startswith('.') and f.lower().endswith('psv'):
                files_names.append(file_path)

    files_names.sort()

    # Saving test data
    for idx in tqdm.tqdm(test_indicies, desc="Saving test data in 'test_data' directory", total=len(test_indicies)):
        patient_file = files_names[idx]
        patient_name = os.path.basename(patient_file)
        patient_data = pd.read_csv(patient_file, delimiter='|')

        output_file_path = os.path.join(test_data_path, patient_name)
        patient_data.to_csv(output_file_path, index=False, sep='|')

    # Metrics
    train_losses, val_losses, test_losses, = metrics['train_loss'], metrics['val_loss'], metrics['test_loss']
    train_accuracies, val_accuracies, test_accuracies = metrics['train_accuracy'], metrics['val_accuracy'], metrics[
        'test_accuracy']

    if 'physionet2019' in destination_path:  # When using Unity

        # Saving Locally
        plot_losses_and_accuracies(train_losses, test_losses, train_accuracies, test_accuracies,
                                   save_path='./data/logs')  # Local

    plot_losses_and_accuracies(train_losses, test_losses, train_accuracies, test_accuracies,
                               save_path=destination_path)
