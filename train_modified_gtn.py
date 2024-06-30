import logging

import torch
import tqdm
from torch import nn, optim

from models.modified_gtn import ModifiedGatedTransformerNetwork, initialize_experiment
from utils.config import gtn_param
from utils.loader import make_loader


def save_model(model, model_name):
    logging.info(f"Saving the model with model_name: {model_name}")
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
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


def load_model_dataparallel(model, model_name):
    print(f"Loading {model_name} GTN model...")
    logging.info(f"Loading GTN model...")
    state_dict = torch.load(model_name)

    # Handle "module." prefix if necessary
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    print(f"Model is set to eval() mode...")
    # logging.info(f"Model is set to eval() mode...")
    model.eval()
    print(f"Model is on the device: {device}")
    # logging.info(f"Model is on the device: {device}")
    model.to(device)

    return model


if __name__ == '__main__':

    training_examples, lengths_list, is_sepsis, writer, destination_path = initialize_experiment('final_dataset.pickle')
    train_loader, test_loader, train_indicies, test_indicies = make_loader(training_examples, lengths_list, is_sepsis,
                                                                           batch_size=1, mode='padding_and_lengths')

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
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []

    epochs = 2
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        correct_train, total_train = 0, 0
        train_loader_tqdm = tqdm.tqdm(enumerate(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for idx, (inputs, labels, lengths) in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _, _, _, _, _, _ = model(inputs.to(torch.float32), lengths, 'train')

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

        save_model(model, model_name=f"single_batch_without_dataparallel_epoch_{epoch}.pkl")

    save_model(model, model_name=f"single_batch_without_dataparallel.pkl")
