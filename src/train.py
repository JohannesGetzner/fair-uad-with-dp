import os
import pandas as pd
os.environ["WANDB__SERVICE_WAIT"] = "300"
import sys
sys.path.append('..')
import yaml
import gc
import wandb
import torch
from opacus.validators import ModuleValidator
from dotmap import DotMap
from datetime import datetime
from opacus import PrivacyEngine
from models.FAE.fae import FeatureReconstructor
from argparse import ArgumentParser, BooleanOptionalAction
from utils.utils import init_wandb, construct_log_dir, seed_everything
from utils.train_utils import train, train_dp, test, DEFAULT_CONFIG, load_data, init_model
import json


parser = ArgumentParser()
parser.add_argument('--run_config', default="dp", type=str)
parser.add_argument('--run_version', default="v1", type=str)
parser.add_argument('--model_type', default="FAE", type=str)
parser.add_argument('--protected_attr_percent', default=None, type=float)
parser.add_argument('--lr', default=None, type=float)
parser.add_argument('--n_adam', action=BooleanOptionalAction, default=None)
parser.add_argument('--group_name_mod', default=None, type=str)
parser.add_argument('--job_type_mod', default=None, type=str)
parser.add_argument('--dataset', default=None, type=str)
parser.add_argument('--test_dataset', default=None, type=str)
parser.add_argument('--train_dataset_mode', default=None, type=str)
parser.add_argument('--loss_weight_type', default=None, type=str)
parser.add_argument('--weight', default=None, type=float)
parser.add_argument('--second_stage_epsilon', default=None, type=float)
parser.add_argument('--second_stage_steps', default=None, type=int)
parser.add_argument('--pretrained_model_path', default=None, type=str)
parser.add_argument('--wb_custom_run_name',  default=None, type=str)
parser.add_argument('--upsampling_strategy', default=None, type=str)
parser.add_argument('--custom_sr', action=BooleanOptionalAction, default=None)
parser.add_argument('--no_img_log', action=BooleanOptionalAction, default=None)
parser.add_argument('--best_and_worst_subsets', action=BooleanOptionalAction, default=None)
parser.add_argument('--effective_dataset_size', default=None, type=float)
parser.add_argument('--hidden_dims', nargs='+', default=None, type=int)
parser.add_argument('--dataset_random_state', default=None, type=int)
parser.add_argument('--n_training_samples', default=None, type=int)
parser.add_argument('--d', type=str, default=str(datetime.strftime(datetime.now(), format="%Y-%m-%d %H:%M:%S")))
DYNAMIC_PARAMS = parser.parse_args()


def num_steps_to_epochs(num_steps, train_loader):
    return num_steps // len(train_loader)


def initialize_configuration(new_config, static_params):
    # update config with static params
    for arg_name, arg_value in static_params.items():
        new_config[arg_name] = arg_value
    # update config with run params
    for arg_name in vars(DYNAMIC_PARAMS).keys():
        if getattr(DYNAMIC_PARAMS, arg_name) is not None:
            new_config[arg_name] = getattr(DYNAMIC_PARAMS, arg_name)
    if new_config.second_stage_epsilon:
        new_config.epsilon = new_config.epsilon - new_config.second_stage_epsilon
    return new_config


def run(config):
    # load data
    train_loader, val_loader, test_loader, max_sample_freq = load_data(config)
    config.epochs = num_steps_to_epochs(config.num_steps, train_loader)
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
        model, optimizer = init_model(config)
        if config.dp:
            # Init DP
            privacy_engine = PrivacyEngine(accountant="rdp")
            config.delta = 1 / len(train_loader.dataset)
            if config.upsampling_strategy:
                if config.custom_sr:
                    config.sample_rate = max_sample_freq/len(train_loader)
                else:
                    config.sample_rate = None
            else:
                config.sample_rate = None
            model, optimizer, dp_train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=config.epsilon,
                target_delta=config.delta,
                max_grad_norm=config.max_grad_norm,
                epochs=config.epochs,
                custom_sample_rate=config.sample_rate
            )
            model, steps_done = train_dp(model, optimizer, dp_train_loader, val_loader, config, log_dir, privacy_engine)
        else:
            model, steps_done = train(model, optimizer, train_loader, val_loader, config, log_dir)
        test(config, model, test_loader, log_dir)

        if config.second_stage_steps:
            _ = run_stage_two(model, optimizer, config, log_dir, steps_done)
        wandb.finish()
        del model
        torch.cuda.empty_cache()
        gc.collect()

def run_dataset_distillation(config):
    train_loaders, val_loader, test_loader, max_sample_freq = load_data(config)
    config.epochs = num_steps_to_epochs(config.num_steps, train_loaders[0])
    for idx, train_loader in enumerate(train_loaders):
        if idx < 3844:
            continue
        config.job_type_mod = f"train_loader_{idx}"
        config.disable_wandb = True
        time_start = datetime.now()
        for i in range(config.num_seeds):
            config.seed = config.initial_seed + i
            # get log dir
            log_dir, group_name, job_type = construct_log_dir(config, current_time)
            # create a dataframe from trainloader.dataset with the labels, meta and filenames
            temp_df = pd.DataFrame({
                'meta': train_loader.dataset.meta,
                'labels': train_loader.dataset.labels,
                'filenames': train_loader.dataset.filenames,
                'index_mapping': train_loader.dataset.index_mapping_cpy,
            })
            # save
            init_wandb(config, log_dir, group_name, job_type)
            # reproducibility
            print(f"Setting seed to {config.seed}...")
            seed_everything(config.seed)
            # create log dir
            if not config.debug:
                os.makedirs(log_dir, exist_ok=True)
            temp_df.to_csv(os.path.join(log_dir, 'train_loader.csv'), index=False)
            # init model
            model, optimizer = init_model(config)
            model, _ = train(model, optimizer, train_loader, val_loader, config, log_dir)
            test(config, model, test_loader, log_dir)
            wandb.finish()
        print(f"Time taken for train_loader_{idx}: {(datetime.now() - time_start).total_seconds()}s")


def run_best_and_worst_subsets(config):
    train_loaders, val_loader, test_loader, max_sample_freq = load_data(config)
    v2 = True
    with open(f'logs_persist/distillation/subsets_{config.model_type}_{config.dataset}.json' if not v2 else f"logs_persist/distillation/subsets_combined_{config.model_type}_{config.dataset}.json", 'r') as f:
        subsets = json.load(f)
    config.epochs = num_steps_to_epochs(config.num_steps, train_loaders[0])
    for idx, train_loader in enumerate(train_loaders):
        subset = subsets[idx]
        if not v2: assert subset["filenames"] == train_loader.dataset.filenames

        tags = ["worst" if subset["mode"] == "min" else "best", str(subset["size"])]
        if not v2: tags += ["young" if subset["score_var"] == "test/lungOpacity_young_subgroupAUROC" else "old"]
        config.job_type_mod = f"train_loader_{idx}_" + "_".join(tags)
        tags = ["distill_" + tag for tag in tags]
        for i in range(config.num_seeds):
            config.seed = config.initial_seed + i
            # get log dir
            log_dir, group_name, job_type = construct_log_dir(config, current_time)
            # create a dataframe from trainloader.dataset with the labels, meta and filenames
            temp_df = pd.DataFrame({'meta': train_loader.dataset.meta, 'labels': train_loader.dataset.labels, 'filenames': train_loader.dataset.filenames})
            # save
            run = init_wandb(config, log_dir, group_name, job_type)
            run.tags = run.tags + tuple(tags)
            # reproducibility
            print(f"Setting seed to {config.seed}...")
            seed_everything(config.seed)
            # create log dir
            if not config.debug:
                os.makedirs(log_dir, exist_ok=True)
            temp_df.to_csv(os.path.join(log_dir, 'train_loader.csv'), index=False)
            # init model
            model, optimizer = init_model(config)
            model, _ = train(model, optimizer, train_loader, val_loader, config, log_dir)
            test(config, model, test_loader, log_dir)
            wandb.finish()
            del model
            torch.cuda.empty_cache()
            gc.collect()
    train_loaders, val_loader, test_loader, max_sample_freq = load_data(config)

def run_stage_two(model, optimizer, config, log_dir, steps_done):
    print("\nStarting second stage...")
    modified_config = config.copy()
    modified_config.initial_stage_protected_attr_percent = modified_config.protected_attr_percent
    modified_config.protected_attr_percent = 0

    if modified_config.dp:
        modified_config.epsilon = modified_config.second_stage_epsilon
        print("Second stage epsilon:", modified_config.epsilon)
    train_loader, val_loader, test_loader, max_sample_freq = load_data(modified_config)
    modified_config.epochs = num_steps_to_epochs(modified_config.second_stage_steps, train_loader)
    if modified_config.dp:
        privacy_engine = PrivacyEngine(accountant="rdp")
        modified_config.delta = 1 / len(train_loader.dataset)
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


def train_on_one_but_test_val_on_other(config):
    config.group_name_mod = f"testOn-{config.test_dataset}-mode-{config.train_dataset_mode}"
    train_dataset = config.dataset
    train_loaders, val_loader, test_loader_A, max_sample_freq = load_data(config)
    config.epochs = num_steps_to_epochs(config.num_steps, train_loaders[0])
    if type(train_loaders) != list:
        train_loaders = [train_loaders]
    config.dataset = config.test_dataset
    config.train_dataset_mode = ""
    _, _, test_loader_B, max_sample_freq = load_data(config)
    config.dataset = train_dataset
    for idx, train_loader in enumerate(train_loaders):
        config.job_type_mod = f"nsamples-{len(train_loader.dataset)}"
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
            model, optimizer = init_model(config)
            model, _ = train(model, optimizer, train_loader, val_loader, config, log_dir)
            test(config, model, test_loader_A, log_dir, file_name_mod=train_dataset)
            test(config, model, test_loader_B, log_dir, file_name_mod=config.test_dataset)
            wandb.finish()


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
    current_config = initialize_configuration(default_config.copy(), run_config)
    print(f"Run config: {run_config}")
    if current_config.n_training_samples:
        run_dataset_distillation(current_config)
    elif current_config.best_and_worst_subsets:
        run_best_and_worst_subsets(current_config)
    elif current_config.test_dataset:
        train_on_one_but_test_val_on_other(current_config)
    else:
        run(current_config)
