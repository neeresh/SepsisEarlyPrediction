import os

import numpy as np
import pandas as pd
import random
import torch

import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(home_path, log_dir, last_model, best_model):
    save_dict = {
        "last": last_model,
        "best": best_model
    }
    # save classification report
    save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
    torch.save(save_dict, save_path)


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(algorithm, test_loader, device):
    feature_extractor = algorithm.feature_extractor.to(device)
    classifier = algorithm.classifier.to(device)

    feature_extractor.eval()
    classifier.eval()

    total_loss, preds_list, labels_list = [], [], []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.float().to(device)
            labels = labels.view((-1)).long().to(device)

            # forward pass
            features = feature_extractor(data)
            predictions = classifier(features)

            # compute loss
            loss = F.cross_entropy(predictions, labels)
            total_loss.append(loss.item())
            pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

            # append predictions and labels
            preds_list.append(pred)
            labels_list.append(labels)

    loss = torch.tensor(total_loss).mean()  # average loss
    full_preds = torch.cat((preds_list))
    full_labels = torch.cat((labels_list))

    return loss, full_preds, full_labels


def calculate_metrics(algorithm, trg_test_dl, dataset_configs, device):
    # Making predictions
    loss, full_preds, full_labels = evaluate(algorithm, trg_test_dl, device)

    # metrics
    num_classes = dataset_configs.num_classes
    ACC = Accuracy(task="multiclass", num_classes=num_classes)
    F1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
    # AUROC = AUROC(task="multiclass", num_classes=num_classes)

    # accuracy
    acc = ACC(full_preds.argmax(dim=1).cpu(), full_labels.cpu()).item()

    # f1
    f1 = F1(full_preds.argmax(dim=1).cpu(), full_labels.cpu()).item()

    # auroc
    # auroc = AUROC(full_preds.cpu(), full_labels.cpu()).item()
    auroc = 0

    return acc, f1, auroc


def calculate_risks(algorithm, src_test_dl, trg_test_dl, few_shot_dl_5, device):
    # calculation based source test data
    loss, full_preds, full_labels = evaluate(algorithm, src_test_dl, device)
    src_risk = loss.item()

    # calculation based few_shot test data
    loss, full_preds, full_labels = evaluate(algorithm, few_shot_dl_5, device)
    fst_risk = loss.item()

    # calculation based target test data
    loss, full_preds, full_labels = evaluate(algorithm, trg_test_dl, device)
    trg_risk = loss.item()

    return src_risk, fst_risk, trg_risk


def append_results_to_tables(table, scenario, run_id, metrics):
    # Create metrics and risks rows
    results_row = [scenario, run_id, *metrics]

    # Create new dataframes for each row
    results_df = pd.DataFrame([results_row], columns=table.columns)

    # Concatenate new dataframes with original dataframes
    table = pd.concat([table, results_df], ignore_index=True)

    return table


def add_mean_std_table(table, columns):
    # Calculate average and standard deviation for metrics
    avg_metrics = [table[metric].mean() for metric in columns[2:]]
    std_metrics = [table[metric].std() for metric in columns[2:]]

    # Create dataframes for mean and std values
    mean_metrics_df = pd.DataFrame([['mean', '-', *avg_metrics]], columns=columns)
    std_metrics_df = pd.DataFrame([['std', '-', *std_metrics]], columns=columns)

    # Concatenate original dataframes with mean and std dataframes
    table = pd.concat([table, mean_metrics_df, std_metrics_df], ignore_index=True)

    # Create a formatting function to format each element in the tables
    format_func = lambda x: f"{x:.4f}" if isinstance(x, float) else x

    # Apply the formatting function to each element in the tables
    table = table.map(format_func)

    return table


def save_tables_to_file(exp_log_dir, table_results, name):
    table_results.to_csv(os.path.join(exp_log_dir, f"{name}.csv"))


def load_checkpoint(model_dir):
    # checkpoint = torch.load(os.path.join(home_path, model_dir, 'checkpoint.pt'))
    checkpoint = torch.load(os.path.join(model_dir, 'checkpoint.pt'))
    last_model = checkpoint['last']
    best_model = checkpoint['best']
    return last_model, best_model


# For DIRT-T
class EMA:
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.params = self.shadow.keys()

    def __call__(self, model):
        if self.decay > 0:
            for name, param in model.named_parameters():
                if name in self.params and param.requires_grad:
                    self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)
                    param.data = self.shadow[name]
