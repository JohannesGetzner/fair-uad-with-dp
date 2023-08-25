import sys
sys.path.append('..')
import yaml
import wandb
from argparse import ArgumentParser
from utils.utils import construct_log_dir, init_wandb
from datetime import datetime
from utils.train_utils import train, train_dp, test, DEFAULT_CONFIG, load_data, init_model
from opacus import PrivacyEngine

import torch
import gc

parser = ArgumentParser()
parser.add_argument('--num_sweeps', default=10, type=float)
parser.add_argument('--protected_attr_percent', default=0.5, type=float)
parser.add_argument('--job_type_mod', default=None, type=str)
parser.add_argument('--group_name_mod', default=None, type=str)
parser.add_argument('--d', type=str, default=str(datetime.strftime(datetime.now(), format="%Y.%m.%d-%H:%M:%S")))
RUN_CONFIG = parser.parse_args()


def main():
    init_wandb(config, log_dir, group_name, job_type, sweep=True)
    for param in sweep_config["parameters"].keys():
        config[param] = wandb.config[param]
    _, model, optimizer = init_model(config)
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
        final_model = train_dp(model, optimizer, data_loader, val_loader, config, log_dir, privacy_engine)
    else:
        final_model = train(model, optimizer, train_loader, val_loader, config, log_dir)
    test(config, final_model, test_loader, log_dir)
    del final_model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    with open('sweep_config.yml', 'r') as f:
        c = yaml.safe_load(f)
        exp_config = c['experiment_config']
        sweep_config = c['sweep_config']
    config = DEFAULT_CONFIG.copy()
    config.seed = config.initial_seed
    for k, v in exp_config.items():
        config[k] = v

    config.protected_attr_percent = RUN_CONFIG.protected_attr_percent
    config.job_type_mod = RUN_CONFIG.job_type_mod
    config.group_name_mod = RUN_CONFIG.group_name_mod

    train_loader, val_loader, test_loader = load_data(config)
    log_dir, group_name, job_type = construct_log_dir(config, RUN_CONFIG.d)
    sweep_config["name"] = group_name
    sweep_id = wandb.sweep(sweep_config, project=config.wandb_project)
    wandb.agent(sweep_id, function=main, count=RUN_CONFIG.num_sweeps)
    wandb.finish()