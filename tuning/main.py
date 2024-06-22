from functools import partial

from ray import tune

from tuning.search_space import get_hyperparameters

from ray.tune.schedulers import ASHAScheduler

from tuning.training import train_sepsis


def main(num_samples=10, max_epochs=10, gpus_per_trial=2):
    config = get_hyperparameters()
    scheduler = ASHAScheduler(metric="loss", mode="min", max_t=max_epochs, grace_period=1, reduction_factor=2)
    result = tune.run(partial(train_sepsis), resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
                      config=config, num_samples=num_samples, scheduler=scheduler)

    best_trial = result.get_best_trial("loss", "min", "last")

    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")


if __name__ == "__main__":
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)

