from ray import tune


def get_hyperparameters():
    hyperparameter_searchspace = {
        # "l1": tune.choice([2 ** i for i in range(9)]),
        # "l2": tune.choice([2 ** i for i in range(9)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
        }
    
    return hyperparameter_searchspace
