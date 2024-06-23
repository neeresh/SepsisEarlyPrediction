from ray import tune


def get_hyperparameters():
    
    return {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
        }
