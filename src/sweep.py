import sys
sys.path.append('..')
import yaml
import wandb
from argparse import ArgumentParser
from utils.utils import construct_log_dir, init_wandb
from datetime import datetime
from utils.train_utils import train, train_dp, test, DEFAULT_CONFIG, load_data


parser = ArgumentParser()
parser.add_argument('--num_sweeps', default=10, type=float)
parser.add_argument('--protected_attr_percent', default=0.5, type=float)
parser.add_argument('--weight', default=0.9, type=float)
parser.add_argument('--job_type_mod', default="", type=str)
parser.add_argument('--d', type=str, default=str(datetime.strftime(datetime.now(), format="%Y.%m.%d-%H:%M:%S")))
RUN_CONFIG = parser.parse_args()


def main():
    init_wandb(config, log_dir, group_name, job_type, sweep=True)
    for param in sweep_config["parameters"].keys():
        config[param] = wandb.config[param]
    if config.dp:
        final_model = train_dp(train_loader, val_loader, config, log_dir)
    else:
        final_model = train(train_loader, val_loader, config, log_dir)
    test(config, final_model, test_loader, log_dir)


if __name__ == '__main__':
    with open('sweep_config.yml', 'r') as f:
        c = yaml.safe_load(f)
        exp_config = c['experiment_config']
        sweep_config = c['sweep_config']
    config = DEFAULT_CONFIG.copy()
    for k, v in exp_config.items():
        config[k] = v
    config.seed = config.initial_seed
    config.protected_attr_percent = RUN_CONFIG.protected_attr_percent
    config.weigh_loss = config.weigh_loss + f"_{RUN_CONFIG.weight}"
    if RUN_CONFIG.job_type_mod != "":
        config.job_type_mod = RUN_CONFIG.job_type_mod

    train_loader, val_loader, test_loader = load_data(config)
    log_dir, group_name, job_type = construct_log_dir(config, RUN_CONFIG.d)
    sweep_config["name"] = group_name
    sweep_id = wandb.sweep(sweep_config, project=config.wandb_project)
    wandb.agent(sweep_id, function=main, count=RUN_CONFIG.num_sweeps)
    wandb.finish()