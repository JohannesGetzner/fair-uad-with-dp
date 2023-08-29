import math
import sys
import yaml
import gc
sys.path.append('..')
import os
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
from models.FAE.fae import FeatureReconstructor
import wandb
import torch
from utils.utils import init_wandb, construct_log_dir, seed_everything
from dotmap import DotMap
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
parser.add_argument('--second_stage_epsilon', default=None, type=float)
parser.add_argument('--second_stage_epochs', default=None, type=int)

parser.add_argument('--pretrained_model_path', default=None, type=str)
parser.add_argument('--wb_custom_run_name',  default=None, type=str)

parser.add_argument('--d', type=str, default=str(datetime.strftime(datetime.now(), format="%Y-%m-%d %H:%M:%S")))
DYNAMIC_PARAMS = parser.parse_args()

""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""


def initialize_configuration(new_config, static_params):
    # update config with static params
    for arg_name, arg_value in static_params.items():
        new_config[arg_name] = arg_value

    # update config with run params
    for arg_name in vars(DYNAMIC_PARAMS).keys():
        if getattr(DYNAMIC_PARAMS, arg_name) is not None:
            new_config[arg_name] = getattr(DYNAMIC_PARAMS, arg_name)
    if new_config.dp:
        new_config.epsilon = new_config.epsilon - new_config.second_stage_epsilon
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
        # create log dir
        if not config.debug:
            os.makedirs(log_dir, exist_ok=True)
        # init model
        if "pretrained_model_path" in config.keys():
            model, steps_done, old_c = load_pretrained_model(config.pretrained_model_path)
            model = model.to(config.device)
            # override config
            for k, v in old_c.items():
                config[k] = v
            _, _, optimizer = init_model(config)
        else:
            _, model, optimizer = init_model(config)
            if config.dp:
                # Init DP
                privacy_engine = PrivacyEngine(accountant="rdp")
                config.delta = 1 / (len(train_loader) * train_loader.batch_size)
                model, optimizer, dp_train_loader = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    target_epsilon=config.epsilon,
                    target_delta=config.delta,
                    max_grad_norm=config.max_grad_norm,
                    epochs=config.epochs
                )
                model, steps_done = train_dp(model, optimizer, dp_train_loader, val_loader, config, log_dir, privacy_engine)
            else:
                model, steps_done = train(model, optimizer, train_loader, val_loader, config, log_dir)
            test(config, model, test_loader, log_dir)

        if config.second_stage_epsilon:
            _ = run_stage_two(model, optimizer, config, log_dir, steps_done)
        wandb.finish()
        del model
        torch.cuda.empty_cache()
        gc.collect()


def run_stage_two(model, optimizer, config, log_dir, steps_done):
    print("\nStarting second stage...")
    modified_config = config.copy()
    modified_config.initial_stage_protected_attr_percent = modified_config.protected_attr_percent
    modified_config.protected_attr_percent = 0
    modified_config.epochs = modified_config.second_stage_epochs
    if modified_config.dp:
        modified_config.epsilon = modified_config.second_stage_epsilon
        print("Second stage epsilon:", modified_config.epsilon)
    train_loader, val_loader, test_loader = load_data(modified_config)
    if modified_config.dp:
        privacy_engine = PrivacyEngine(accountant="rdp")
        modified_config.delta = 1 / (len(train_loader) * train_loader.batch_size)
        model.train()
        model, optimizer, dp_train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=modified_config.epsilon,
            target_delta=modified_config.delta,
            max_grad_norm=modified_config.max_grad_norm,
            epochs=modified_config.epochs
        )
        model, _ = train_dp(model, optimizer, dp_train_loader, val_loader, modified_config, log_dir, privacy_engine, prev_step=steps_done)
    else:
        model, _ = train(model, optimizer, train_loader, val_loader, modified_config, log_dir, prev_step=steps_done)
    test(modified_config, model, test_loader, log_dir, stage_two=True)
    return model


def load_pretrained_model(path):
    # load model from logs
    path = os.path.join(os.getcwd(), 'logs_persist', path)
    checkpoint = torch.load(path)
    old_config = DotMap(checkpoint["config"]['_map'])
    if "loss_weight_type" not in old_config.keys():
        old_config.loss_weight_type = None
    model = FeatureReconstructor(old_config)
    state_dict = checkpoint["model"]
    new_state_dict = {key.replace('_module.', ''): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)
    return model, checkpoint["step"], old_config


if __name__ == '__main__':
    # get time
    current_time = DYNAMIC_PARAMS.d
    # copy default config
    default_config = DEFAULT_CONFIG.copy()
    # get run config
    with open(f'run_{DYNAMIC_PARAMS.run_config}_config.yml', 'r') as f:
        run_configs = yaml.safe_load(f)
    run_config = run_configs[DYNAMIC_PARAMS.run_version]
    print(f"Running {DYNAMIC_PARAMS.run_config}/{DYNAMIC_PARAMS.run_version}...")
    print(f"Run config: {run_config}")
    current_config = initialize_configuration(default_config.copy(), run_config)
    run(current_config)
