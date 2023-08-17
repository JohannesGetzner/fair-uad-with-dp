import sys
import yaml
sys.path.append('..')
import os
from argparse import ArgumentParser
from datetime import datetime
import wandb


from src.utils.utils import init_wandb, construct_log_dir
from opacus.validators.utils import register_module_fixer
from torch import nn
from utils.train_utils import train, train_dp, test, DEFAULT_CONFIG, load_data


@register_module_fixer([nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm])
def _batchnorm_to_groupnorm(module) -> nn.GroupNorm:
    num_groups = 32
    if module.num_features % num_groups != 0:
        num_groups = 25
    return nn.GroupNorm(min(num_groups, module.num_features), module.num_features, affine=module.affine)


""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""

parser = ArgumentParser()
parser.add_argument('--run_name', default="dp_1", type=str)
parser.add_argument('--job_type_mod', default="", type=str)
parser.add_argument('--protected_attr_percent', default=0.5, type=float)
parser.add_argument('--custom', default=0.9, type=float)
parser.add_argument('--d', type=str, default=str(datetime.strftime(datetime.now(), format="%Y.%m.%d-%H:%M:%S")))
RUN_CONFIG = parser.parse_args()

""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""

def run(config, run_config):
    # update default values config
    # replace the default config values with the run config
    for arg_name, arg_value in run_config.items():
        config[arg_name] = arg_value
    # get protected attribute percent value
    config.protected_attr_percent = RUN_CONFIG.protected_attr_percent
    config.weigh_loss = config.weigh_loss + f"_{RUN_CONFIG.custom}"
    if RUN_CONFIG.job_type_mod != "":
        config.job_type_mod = RUN_CONFIG.job_type_mod
    # load data
    train_loader, val_loader, test_loader = load_data(config)
    # iterate over seeds
    config.seed = config.initial_seed
    for i in range(config.num_seeds):
        config.seed = config.initial_seed + i
        # get log dir
        log_dir, group_name, job_type = construct_log_dir(config, current_time)
        init_wandb(config, log_dir, group_name, job_type)
        # create log dir
        if not config.debug:
            os.makedirs(log_dir, exist_ok=True)
        if config.dp:
            final_model = train_dp(train_loader, val_loader, config, log_dir)
        else:
            final_model = train(train_loader, val_loader, config, log_dir)
        test(config, final_model, test_loader, log_dir)
        wandb.finish()


if __name__ == '__main__':
    # get time
    current_time = RUN_CONFIG.d
    # copy default config
    new_config = DEFAULT_CONFIG.copy()
    # set initial seed
    new_config.seed = new_config.initial_seed
    # get run config
    with open('runs_config.yml', 'r') as f:
        run_configs = yaml.safe_load(f)
    # choose runs to execute or run all
    run_config = run_configs[RUN_CONFIG.run_name]
    print(f"Running {RUN_CONFIG.run_name}")
    run(new_config.copy(), run_config)
