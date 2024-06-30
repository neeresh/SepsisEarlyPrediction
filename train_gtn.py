import datetime
import logging
import os
from typing import Optional

import pandas as pd
import torch
import tqdm
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader

from models.gtn import GatedTransformerNetwork
from utils.config import gtn_param
from utils.loader import make_loader
from utils.path_utils import project_root
from utils.plot_metrics import plot_losses_and_accuracies

device = 'cuda'


def _setup_destination(current_time):
    log_path = os.path.join(project_root(), 'data', 'logs', current_time)
    os.mkdir(log_path)
    logging.basicConfig(filename=os.path.join(log_path, current_time + '.log'), level=logging.DEBUG)

    return log_path


def _log(message: str = '{}', value: any = None):
    print(message.format(value))
    logging.info(message.format(value))


def initialize_experiment(data_file=None):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    destination_path = _setup_destination(current_time)

    if data_file is None:
        data_file = "training_ffill_bfill_zeros.pickle"

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
    if class_0_weight is not None and class_1_weight is not None:
        print(f"Using manual weights for classes 0 and 1")
        logging.info(f"Class0 weight: {class_0_weight} & Class1 weight: {class_1_weight}")
        manual_weights = torch.tensor([class_0_weight, class_1_weight]).to(device)
        criterion = nn.CrossEntropyLoss(weight=manual_weights)

    else:
        criterion = nn.CrossEntropyLoss()

    # GTN
    optimizer = optim.Adagrad(model.parameters(), lr=1e-4)  # GTN

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer)

    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        correct_train, total_train = 0, 0

        train_loader_tqdm = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for inputs, labels in train_loader_tqdm:
            optimizer.zero_grad()

            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _, _, _, _, _, _ = model(inputs.to(torch.float32), 'train')
            loss = criterion(outputs, labels)

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

        # Validation phase
        if val_loader:
            model.eval()
            running_val_loss = 0.0
            correct_val, total_val = 0, 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs, _, _, _, _, _, _ = model(inputs.to(torch.float32), 'test')
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            epoch_val_accuracy = correct_val / total_val
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_accuracy)
        else:
            epoch_val_loss = "N/A"
            epoch_val_accuracy = "N/A"

        # Testing phase
        model.eval()
        running_test_loss = 0.0
        correct_test, total_test = 0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _, _, _, _, _, _ = model(inputs.to(torch.float32), 'test')
                loss = criterion(outputs, labels)
                running_test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        epoch_test_accuracy = correct_test / total_test
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_accuracy)

        # scheduler.step()

        message = f"Epoch {epoch + 1}/{epochs} - " \
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}, " \
                  f"Val Loss: {epoch_val_loss}, Val Acc: {epoch_val_accuracy}, " \
                  f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_accuracy:.4f}"
        tqdm.tqdm.write(message)
        logging.info(message)

    # Saving the model
    save_model(model, model_name="reduced_model.pkl")

    return {"train_loss": train_losses, "val_loss": val_losses if val_loader else None, "test_loss": test_losses,
            "train_accuracy": train_accuracies, "val_accuracy": val_accuracies if val_loader else None,
            "test_accuracy": test_accuracies}


def save_model(model, model_name="model_gtn.pkl"):
    logging.info(f"Saving the model with model_name: {model_name}")
    torch.save(model.state_dict(), model_name)
    logging.info(f"Saving successfull!!!")


def load_model(model, model_name="model_gtn.pkl"):
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

    # Getting Data and Loaders
    data_file = "final_dataset.pickle"
    training_examples, lengths_list, is_sepsis, writer, destination_path = initialize_experiment(data_file)

    # Default
    class_0_weight = 40336 / (37404 * 2)  # 37404
    class_1_weight = 40336 / (2932 * 2)
    train_loader, test_loader, train_indicies, test_indicies = make_loader(training_examples, lengths_list, is_sepsis,
                                                                           batch_size=16, mode='padding')

    # Reducing the samples to have balanced dataset
    # class_0_weight = 8543 / (5611 * 2)
    # class_1_weight = 8543 / (2932 * 2)

    # sepsis = pd.Series(is_sepsis)
    # positive_sepsis_idxs = sepsis[sepsis == 1].index
    # negative_sepsis_idxs = sepsis[sepsis == 0].sample(frac=0.15).index
    # all_samples = list(positive_sepsis_idxs) + list(negative_sepsis_idxs)
    #
    # print(f"Total samples: {len(all_samples)}")
    # train_indicies, test_indicies = train_test_split(all_samples, test_size=0.2, random_state=42)
    # train_loader, test_loader, train_indicies, test_indicies = make_loader(training_examples, lengths_list, is_sepsis,
    #                                                                        128, 'padding',
    #                                                                        num_workers=8, train_indicies=train_indicies,
    #                                                                        test_indicies=test_indicies)

    config = gtn_param
    (d_input, d_channel), d_output = train_loader.dataset.data[0].shape, 2  # (time_steps, features, num_classes)
    print(f"d_input: {d_input}, d_channel: {d_channel}, d_output: {d_output}")
    num_epochs = 5

    print(d_input, d_channel, d_output)

    logging.info(config)
    logging.info(f"d_input: {d_input}, d_channel: {d_channel}, d_output: {d_output}")
    logging.info(f"Number of epochs: {num_epochs}")

    model = GatedTransformerNetwork(d_model=config['d_model'], d_input=d_input, d_channel=d_channel,
                                    d_output=d_output, d_hidden=config['d_hidden'], q=config['q'],
                                    v=config['v'], h=config['h'], N=config['N'], dropout=config['dropout'],
                                    pe=config['pe'], mask=config['mask'], device=device).to(device)

    # metrics = train_model(model, train_loader, test_loader, class_0_weight=class_0_weight,
    #                       class_1_weight=class_1_weight, epochs=num_epochs)

    metrics = train_model(model, train_loader, test_loader, epochs=num_epochs)

    train_losses, val_losses, test_losses, = metrics['train_loss'], metrics['val_loss'], metrics['test_loss']
    train_accuracies, val_accuracies, test_accuracies = metrics['train_accuracy'], metrics['val_accuracy'], metrics[
        'test_accuracy']

    if 'physionet2019' in destination_path:  # When using Unity
        # Saving in my space
        save_path = './data/logs'
        plot_losses_and_accuracies(train_losses, test_losses, train_accuracies, test_accuracies,
                                   save_path=save_path)  # Local

    plot_losses_and_accuracies(train_losses, test_losses, train_accuracies, test_accuracies,
                               save_path=destination_path)
