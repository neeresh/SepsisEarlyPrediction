import collections

import os

from models.adatime.da.models import get_backbone_class
from models.adatime.da.algorithms import get_algorithm_class
from models.adatime.load_data import load_data
from models.adatime.log_utils.log_experiment import starting_logs
from models.adatime.utils import AverageMeter, save_checkpoint
from utils.model_size import get_model_size
from utils.path_utils import project_root

from models.adatime.configs.get_configs import Config

if __name__ == '__main__':

    # Args
    config = Config()

    # Source and target dataset paths
    source_path = os.path.join(project_root(), 'data', 'tl_datasets', 'pretrain', 'pretrain.pt')
    target_path = os.path.join(project_root(), 'data', 'tl_datasets', 'finetune', 'finetune.pt')

    source_dataloader = load_data(source_path, config)
    target_dataloader = load_data(target_path, config)

    # Domain Adaptation Algorithm
    da_algorithm_name = 'CoTMix'
    da_algorithm = get_algorithm_class(da_algorithm_name)

    # Setting up backbone
    backbone_name = 'GTN'
    da_backbone = get_backbone_class(backbone_name)

    # Initializing algorithm
    device = 'cuda'
    algorithm = da_algorithm(backbone=da_backbone, configs=config, device=device)

    algorithm.to(device)

    # Model Size
    get_model_size(algorithm)

    # Training DA algorithm
    home_path = os.path.join(project_root(), 'results', 'adatime')
    save_dir = f'{da_algorithm_name}'
    dataset_name = 'pretrain_finetune'
    experiment_description = dataset_name
    exp_name = f'{da_algorithm_name}_gtn'
    run_description = f"{da_algorithm_name}_{exp_name}"
    exp_log_dir = os.path.join(home_path, save_dir, experiment_description, f"{run_description}")
    logger, scenario_log_dir = starting_logs(dataset_name, da_algorithm_name,
                                             exp_log_dir, '0', '1', '0')

    loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
    last_model, best_model = algorithm.update(source_dataloader, target_dataloader, loss_avg_meters, logger)

    # Save checkpoint
    save_checkpoint(home_path, scenario_log_dir, last_model, best_model)
