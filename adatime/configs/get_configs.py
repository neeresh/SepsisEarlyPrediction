from adatime.configs.data_model_configs import get_dataset_class
from adatime.configs.hparams import get_hparams_class


def get_configs(dataset_name):
    dataset_class = get_dataset_class(dataset_name)
    hparams_class = get_hparams_class(dataset_name)

    return dataset_class(), hparams_class()
