import os
import pickle
from functools import partial
from pathlib import Path

from ray import tune

import torch.nn as nn
from ray.tune.schedulers import ASHAScheduler

from models.custom_models.gtn import GatedTransformerNetwork
# from tuning.training import train_sepsis, test_accuracy
from tuning.training import tune_sepsis
from utils.path_utils import project_root


def main(num_samples, max_epochs, gpus_per_trial):

    hyperparameters = {
        "d_model": tune.choice([512, 1024]),
        "d_hidden": tune.choice([512, 1024]),
        "q": tune.choice([6, 8, 10]),
        "v": tune.choice([6, 8, 10]),
        "h": tune.choice([6, 8, 10]),
        "N": tune.choice([6, 8, 10]),
        "dropout": tune.loguniform(0.2, 0.5),
        "lr": tune.loguniform(1e-7, 1e-3),
        "batch_size": tune.choice([64, 128, 256]),
        # 'w1': tune.loguniform(0.001, 6),
        # 'w2': tune.loguniform(0.001, 6),
        'epochs': tune.choice([1, 1]),
        # 'labelsmoothing': tune.loguniform(0.0001, 0.05),
        'majority_samples': tune.loguniform(0.1, 0.40),
        }

    scheduler = ASHAScheduler(metric="loss", mode="min", max_t=max_epochs, grace_period=1, reduction_factor=2)

    # output_dir = os.path.join(project_root(), '..', '..', 'results', 'physionet2019', 'hyperparameter_tuning')
    output_dir = os.path.join(project_root(), 'results', 'physionet2019', 'hyperparameter_tuning')

    result = tune.run(partial(tune_sepsis), resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
                      config=hyperparameters, num_samples=num_samples, scheduler=scheduler,
                      storage_path=output_dir)

    best_trial = result.get_best_trial("loss", "min", "last")

    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    # # Testing the best model with best hyperparameters
    # d_input, d_channel, d_output = 336, 63, 2
    # device = "cuda:0"
    # best_trained_model = GatedTransformerNetwork(d_model=best_trial.config['d_model'], d_input=d_input,
    #                                              d_channel=d_channel,
    #                                              d_output=d_output, d_hidden=best_trial.config['d_hidden'],
    #                                              q=best_trial.config['q'],
    #                                              v=best_trial.config['v'], h=best_trial.config['h'], N=best_trial.config['N'],
    #                                              dropout=best_trial.config['dropout'], pe=True,
    #                                              mask=True,
    #                                              device=device).to(device)

    # if gpus_per_trial > 1:
    #     best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)

    # best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="accuracy", mode="max")
    # with best_checkpoint.as_directory() as checkpoint_dir:
    #     data_path = Path(checkpoint_dir) / "data.pkl"
    #     with open(data_path, "rb") as fp:
    #         best_checkpoint_data = pickle.load(fp)

    #     best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
    #     test_acc = test_accuracy(best_trained_model, device)
    #     print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    main(num_samples=1, max_epochs=2, gpus_per_trial=1)
