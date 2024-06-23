from functools import partial

from ray import tune

from tuning.search_space import get_hyperparameters

from ray.tune.schedulers import ASHAScheduler

from tuning.training import train_sepsis


def main(num_samples, max_epochs, gpus_per_trial):
    """
    num_samples: Number of times to sample from hyperparameter space
    """

    config = {"lr": tune.loguniform(1e-4, 1e-1), "batch_size": tune.choice([2, 4, 8, 16])}

    scheduler = ASHAScheduler(metric="loss", mode="min", max_t=max_epochs, grace_period=1, reduction_factor=2)
    result = tune.run(partial(train_sepsis), resources_per_trial={"cpu": 6, "gpu": gpus_per_trial},
                      config=config, num_samples=num_samples, scheduler=scheduler)

    best_trial = result.get_best_trial("loss", "min", "last")

    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")


if __name__ == "__main__":
    main(num_samples=16, max_epochs=5, gpus_per_trial=0.5)
