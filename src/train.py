import sys
import yaml
import gc
sys.path.append('..')
import os
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
import wandb
import torch
from utils.utils import init_wandb, construct_log_dir, seed_everything

from opacus import PrivacyEngine
from utils.train_utils import train, train_dp, test, DEFAULT_CONFIG, load_data, init_model


""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""

parser = ArgumentParser()
parser.add_argument('--run_config', default="dp", type=str)
parser.add_argument('--run_version', default="v1", type=str)

parser.add_argument('--protected_attr_percent', default=None, type=float)
parser.add_argument('--lr', default=None, type=float)
parser.add_argument('--n_adam', action=BooleanOptionalAction, default=None)

parser.add_argument('--group_name_mod', default=None, type=str)
parser.add_argument('--job_type_mod', default=None, type=str)
parser.add_argument('--loss_weight_type', default=None, type=str)
parser.add_argument('--weight', default=None, type=float)
parser.add_argument('--second_stage_epsilon', default=0, type=float)
parser.add_argument('--d', type=str, default=str(datetime.strftime(datetime.now(), format="%Y.%m.%d-%H:%M:%S")))
RUN_ARGS = parser.parse_args()

""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""


def initialize_configuration(new_config, static_params):
    # update config with static params
    for arg_name, arg_value in static_params.items():
        new_config[arg_name] = arg_value

    # update config with run params
    for arg_name in RUN_ARGS.keys():
        if getattr(RUN_ARGS, arg_name) is not None:
            new_config[arg_name] = getattr(RUN_ARGS, arg_name)
    new_config.epsilon = new_config.epsilon - RUN_ARGS.second_stage_epsilon
    return new_config


def run(config):
    # load data
    train_loader, val_loader, test_loader = load_data(config)
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

        if RUN_ARGS.second_stage_epsilon != 0:
            _ = run_stage_two(model, optimizer, config, log_dir)
        wandb.finish()
        del model
        torch.cuda.empty_cache()
        gc.collect()


def run_stage_two(model, optimizer, config, log_dir):
    print("Starting stage two...")
    modified_config = config.copy()
    modified_config.protected_attr_percent = 0
    modified_config.epochs = modified_config.epochs / 3
    modified_config.epsilon = RUN_ARGS.second_stage_epsilon
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


if __name__ == '__main__':
    # get time
    current_time = RUN_ARGS.d
    # copy default config
    default_config = DEFAULT_CONFIG.copy()
    # get run config
    with open(f'run_{RUN_ARGS.run_config}_config.yml', 'r') as f:
        run_configs = yaml.safe_load(f)
    run_config = run_configs[RUN_ARGS.run_version]
    print(f"Running {RUN_ARGS.run_config}/{RUN_ARGS.run_version}...")
    current_config = initialize_configuration(default_config.copy(), run_config)
    run(current_config)
