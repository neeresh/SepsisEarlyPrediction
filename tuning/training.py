from pathlib import Path

import torch
from ray import train
from ray.train import get_checkpoint, Checkpoint
import ray.cloudpickle as pickle
from torch import optim, nn

from models.gtn import GatedTransformerNetwork
from tuning.custom_dataset import get_starters, load_data
from utils.config import gtn_param

import tempfile


# train_indicies, val_indices, test_indicies, examples, lengths_list, is_sepsis = get_starters()
# train_dataset, val_dataset, test_dataset = load_data(train_indicies, val_indices, test_indicies, examples, lengths_list, is_sepsis)

def train_sepsis(hyperparameters=None):

    train_indicies, test_indicies, examples, lengths_list, is_sepsis = get_starters(
        fraction=hyperparameters['majority_samples'])
    train_dataset, val_dataset = load_data(train_indicies, 'NAN', test_indicies, examples, lengths_list, is_sepsis)

    device = "cuda:0"

    # Model
    config = gtn_param
    d_input, d_channel, d_output = 336, 191, 2
    model = GatedTransformerNetwork(d_model=hyperparameters['d_model'], d_input=d_input, d_channel=d_channel,
                                    d_output=d_output, d_hidden=hyperparameters['d_hidden'], q=hyperparameters['q'],
                                    v=hyperparameters['v'], h=hyperparameters['h'], N=hyperparameters['N'],
                                    dropout=hyperparameters['dropout'], pe=config['pe'], mask=hyperparameters['mask'],
                                    device=device).to(device)
    
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    # Criteria, Optimizer, & Scheduler
    manual_weights = torch.tensor([hyperparameters['w1'], hyperparameters['w2']]).to(device)
    criterion = nn.CrossEntropyLoss(weight=manual_weights, label_smoothing=hyperparameters['labelsmoothing'])
    optimizer = optim.Adagrad(model.parameters(), lr=hyperparameters['lr'])  # GTN
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=hyperparameters['epochs'])

    # Checkpoint
    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    # Loading datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(hyperparameters['batch_size']),
                                               shuffle=True, num_workers=8)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(hyperparameters['batch_size']),
                                             shuffle=False, num_workers=8)

    # Training the model
    for epoch in range(start_epoch, hyperparameters['epochs']):
        model.train()
        running_train_loss = 0.0
        epoch_steps = 0
        correct_train, total_train = 0, 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs, _, _, _, _, _, _ = model(inputs.to(torch.float32), 'train')
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # printing statistics
            running_train_loss += loss.item()
            epoch_steps += 1

            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f"% (epoch + 1, i + 1, running_train_loss / epoch_steps))
                running_train_loss = 0.0

        # Validation phase
        running_val_loss = 0.0
        val_steps = 0
        total_val, correct_val = 0, 0

        for i, (inputs, labels) in enumerate(val_loader):
            with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _, _, _, _, _, _ = model(inputs.to(torch.float32), 'test')
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                val_steps += 1

        checkpoint_data = {"epoch": epoch, "net_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report({"loss": running_val_loss / val_steps, "accuracy": correct_val / total_val},
                         checkpoint=checkpoint)

        scheduler.step()

    print("Finished Training")


# def test_accuracy(model, device='cpu'):
#
#     # Testing phase
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=8)
#
#     correct_test, total_test = 0, 0
#     for inputs, labels in test_loader:
#         with torch.no_grad():
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs, _, _, _, _, _, _ = model(inputs.to(torch.float32), 'test')
#             _, predicted = torch.max(outputs, 1)
#             total_test += labels.size(0)
#             correct_test += (predicted == labels).sum().item()
#
#     return correct_test / total_test
