import sys
import yaml
import gc
sys.path.append('..')
import os
from argparse import ArgumentParser
from datetime import datetime
import wandb
import torch
from utils.utils import init_wandb, construct_log_dir
from opacus.validators.utils import register_module_fixer
from torch import nn
from opacus import PrivacyEngine
from utils.train_utils import train, train_dp, test, DEFAULT_CONFIG, load_data, seed_everything, init_model


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
parser.add_argument('--weight', default=0.9, type=float)
parser.add_argument(
    '--d',
    type=str,
    default=str(datetime.strftime(datetime.now(), format="%Y.%m.%d-%H:%M:%S"))
)
parser.add_argument('--stage_two_epsilon', default=0, type=float)
RUN_CONFIG = parser.parse_args()

""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""


def run_stage_two(model, optimizer, config, log_dir):
    print("Starting stage two...")
    modified_config = config.copy()
    modified_config.protected_attr_percent = 0
    modified_config.epochs = modified_config.epochs / 3
    modified_config.epsilon = RUN_CONFIG.stage_two_epsilon
    train_loader, val_loader, test_loader = load_data(modified_config)
    if modified_config.dp:
        privacy_engine = PrivacyEngine(accountant="rdp")
        modified_config.delta = 1 / (len(train_loader) * train_loader.batch_size)
        model.train()
        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=modified_config.epsilon,
            target_delta=modified_config.delta,
            max_grad_norm=modified_config.max_grad_norm,
            epochs=modified_config.epochs
        )
        model = train_dp(model, optimizer, data_loader, val_loader, modified_config, log_dir, privacy_engine)
    else:
        model = train(model, optimizer, train_loader, val_loader, modified_config, log_dir)
    test(modified_config, model, test_loader, log_dir, stage_two=True)
    return model


def run(config, run_config):
    # update default values config
    # replace the default config values with the run config
    for arg_name, arg_value in run_config.items():
        config[arg_name] = arg_value
    # get protected attribute percent value
    config.protected_attr_percent = RUN_CONFIG.protected_attr_percent
    config.weigh_loss = config.weigh_loss + f"_{RUN_CONFIG.weight}"
    if RUN_CONFIG.job_type_mod != "":
        config.job_type_mod = RUN_CONFIG.job_type_mod
    if RUN_CONFIG.stage_two_epsilon != 0:
        config.epsilon = config.epsilon - RUN_CONFIG.stage_two_epsilon
    # load data
    train_loader, val_loader, test_loader = load_data(config)
    # iterate over seeds
    config.seed = config.initial_seed
    for i in range(config.num_seeds):
        config.seed = config.initial_seed + i
        # get log dir
        log_dir, group_name, job_type = construct_log_dir(config, current_time)
        init_wandb(config, log_dir, group_name, job_type)
        # reproducibility
        print(f"Setting seed to {config.seed}...")
        seed_everything(config.seed)
        # Init model
        _, model, optimizer = init_model(config)
        # create log dir
        if not config.debug:
            os.makedirs(log_dir, exist_ok=True)
        if config.dp:
            # Init DP
            privacy_engine = PrivacyEngine(accountant="rdp")
            config.delta = 1 / (len(train_loader) * train_loader.batch_size)
            model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=config.epsilon,
                target_delta=config.delta,
                max_grad_norm=config.max_grad_norm,
                epochs=config.epochs
            )
            model = train_dp(model, optimizer, data_loader, val_loader, config, log_dir, privacy_engine)
        else:
            model = train(model, optimizer, train_loader, val_loader, config, log_dir)
        test(config, model, test_loader, log_dir)

        if RUN_CONFIG.stage_two_epsilon != 0:
            model = run_stage_two(model, optimizer, config, log_dir)

        wandb.finish()
        del model
        torch.cuda.empty_cache()
        gc.collect()


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
