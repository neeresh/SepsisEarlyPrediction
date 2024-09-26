from pathlib import Path

import torch
from ray import train
from ray.train import get_checkpoint, Checkpoint
import ray.cloudpickle as pickle
from torch import optim, nn

# from da.custom_models.gtn import GatedTransformerNetwork
from models.custom_models.gtn_mask import MaskedGatedTransformerNetwork
from tuning.custom_dataset import get_starters, load_data
from utils.config import gtn_param

import tempfile


def tune_sepsis(hyperparameters=None):

    # Get train, test indices and required files.
    train_indices, val_indices, test_indices, examples, lengths_list, is_sepsis = get_starters(
        majority_class=hyperparameters['majority_samples'], data_file="final_dataset.pickle", include_val=False)

    # Get datasets
    train_loader, val_loader, test_loader = load_data(train_indices, val_indices, test_indices,
                                                         examples, lengths_list, is_sepsis,
                                                         hyperparameters["batch_size"])

    device = "cuda"
    config = gtn_param
    d_input, d_channel, d_output = 336, 191, 2
    model = MaskedGatedTransformerNetwork(d_model=hyperparameters['d_model'], d_input=d_input, d_channel=d_channel,
                                    d_output=d_output, d_hidden=hyperparameters['d_hidden'], q=hyperparameters['q'],
                                    v=hyperparameters['v'], h=hyperparameters['h'], N=hyperparameters['N'],
                                    dropout=hyperparameters['dropout'], pe=config['pe'], mask=config['mask'],
                                    device=device)
    
    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters(), lr=hyperparameters['lr'])  # GTN
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=hyperparameters['epochs'])  # Not in GTN Implementation

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

    # Training Phase
    for epoch in range(hyperparameters["epochs"]):

        model.train()
        for idx, (inputs, labels, padded_masks) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs, _, _, _, _, _, _ = model(inputs.to(device).to(torch.float32), 'train', padded_masks)
            loss = criterion(outputs, labels.to(device))

            loss.backward()
            optimizer.step()

        # Validation phase
        running_val_loss = 0.0
        val_steps = 0
        total_val, correct_val = 0, 0

        with torch.no_grad():
            model.eval()
            for idx, (inputs, labels, padded_masks) in enumerate(test_loader):
                outputs, _, _, _, _, _, _ = model(inputs.to(device).to(torch.float32), 'test', padded_masks)
                loss = criterion(outputs, labels.to(device))
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted.detach().cpu() == labels).sum().item()

                running_val_loss += loss.item() * inputs.size(0)
                val_steps += 1

        checkpoint_data = {"epoch": epoch, "net_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report({"loss": running_val_loss / len(test_loader.dataset), "accuracy": correct_val / total_val},
                         checkpoint=checkpoint)

        # scheduler.step()

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
