import functools
import sys
import time

import numpy as np
import yaml

from opacus.utils.batch_memory_manager import BatchMemoryManager

sys.path.append('..')
import os
from argparse import ArgumentParser, BooleanOptionalAction
from collections import defaultdict
from datetime import datetime, timedelta
from dotmap import DotMap
from time import time

import pandas as pd
import torch
import wandb
import math

from src.data.datasets import get_dataloaders
from src.models.DeepSVDD.deepsvdd import DeepSVDD
from src.models.FAE.fae import FeatureReconstructor
from src.utils.metrics import AvgDictMeter, build_metrics
from src.utils.utils import seed_everything, save_checkpoint, init_wandb, construct_log_dir
from opacus import PrivacyEngine
from opacus.validators.utils import register_module_fixer
from torch import nn
from opacus.validators import ModuleValidator


@register_module_fixer([nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm])
def _batchnorm_to_groupnorm(module) -> nn.GroupNorm:
    num_groups = 32
    if module.num_features % num_groups != 0:
        num_groups = 25
    return nn.GroupNorm(min(num_groups, module.num_features), module.num_features, affine=module.affine)


""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""

DEFAULT_CONFIG = {
    # General script settings
    "initial_seed": 1,
    "num_seeds": 1,
    "debug": False,
    "disable_wandb": False,
    "wandb_project": "unsupervised-fairness",
    # Experiment settings
    "experiment_name": "insert-experiment-name",
    # Data settings
    "dataset": "rsna",
    "protected_attr": None,
    "protected_attr_percent": 0.5,
    "img_size": 128,
    "num_workers": 0,
    # Logging settings
    "val_frequency": 200,
    "val_steps": 50,
    "log_frequency": 100,
    "log_img_freq": 1000,
    "num_imgs_log": 8,
    "log_dir": os.path.join('logs', datetime.strftime(datetime.now(), format="%Y.%m.%d-%H:%M:%S")),
    # Hyperparameters
    "lr": 2e-4,
    "weight_decay": 0.0,
    "max_steps": 8000,
    "batch_size": 32,
    # Model settings
    "model_type": "FAE",
    # FAE settings
    "hidden_dims": [100, 150, 200, 300],
    "dropout": 0.1,
    "loss_fn": "ssim",
    "extractor_cnn_layers": ["layer0", "layer1", "layer2"],
    "keep_feature_prop": 1.0,
    # DeepSVDD settings
    "repr_dim": 256,
    # DP settings
    "dp": False,
    "epsilon": 8.0,
    "delta": None,
    "max_grad_norm": 1.0,
    # Other
    "group_name_mod": None,
    "job_type_mod": None,
    "max_physical_batch_size": 512,
}
DEFAULT_CONFIG = DotMap(DEFAULT_CONFIG)

parser = ArgumentParser()
parser.add_argument('--run_all', action=BooleanOptionalAction, default=False)
parser.add_argument('--run_name', default="dp_1", type=str)
parser.add_argument('--sweep', action=BooleanOptionalAction, default=False)
parser.add_argument('--reverse', action=BooleanOptionalAction, default=False)
RUN_CONFIG = parser.parse_args()

DEFAULT_CONFIG.device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {DEFAULT_CONFIG.device}")
if DEFAULT_CONFIG.debug:
    DEFAULT_CONFIG.num_workers = 0
    DEFAULT_CONFIG.max_steps = 1
    DEFAULT_CONFIG.val_frequency = 1
    DEFAULT_CONFIG.val_steps = 1
    DEFAULT_CONFIG.log_frequency = 1
    DEFAULT_CONFIG.batch_size = 8

""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""


def load_data(config):
    print("Loading data...")
    t_load_data_start = time()
    train_loader, val_loader, test_loader = get_dataloaders(dataset=config.dataset, batch_size=config.batch_size,
                                                            img_size=config.img_size, num_workers=config.num_workers,
                                                            protected_attr=config.protected_attr,
                                                            male_percent=config.protected_attr_percent,
                                                            old_percent=config.protected_attr_percent, )
    print(f'Loaded datasets in {time() - t_load_data_start:.2f}s')
    return train_loader, val_loader, test_loader


""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""


def init_model(config):
    print("Initializing model...")
    if config.model_type == 'FAE':
        model = FeatureReconstructor(config)
    elif config.model_type == 'DeepSVDD':
        model = DeepSVDD(config)
    else:
        raise ValueError(f'Unknown model type {config.model_type}')
    model = model.to(config.device)

    # perform model surgery if DP is enabled
    if config.dp:
        model = ModuleValidator.fix(model)

    compiled_model = model  # torch.compile(model)

    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    return model, compiled_model, optimizer


""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""

def compute_mean_per_sample_gradient_norm(model, num_samples):
    mean_per_sample_grad_norm = torch.zeros(num_samples, 1)
    c = 0
    for p in model.parameters():
        if p.grad is not None:
            per_sample_grad = p.grad_sample.detach().clone()
            # collapse tensor to all but the first dimension
            per_sample_grad = per_sample_grad.view(per_sample_grad.shape[0], -1)
            norm = torch.norm(per_sample_grad, dim=1, keepdim=True).to("cpu")
            mean_per_sample_grad_norm += norm
            c += 1
    return mean_per_sample_grad_norm / c


def train_step(model, optimizer, x, y, device):
    model.train()
    optimizer.zero_grad()
    x = x.to(device)
    y = y.to(device)
    loss_dict = model.loss(x, y=y)
    loss = loss_dict['loss']
    loss.backward()
    optimizer.step()
    return loss_dict


def train_step_dp(model, optimizer, x, y, device):
    model.train()
    optimizer.zero_grad()
    x = x.to(device)
    y = y.to(device)
    # TODO: find a better way to compute the loss when wrapped with DP (don't access protected attribute)
    loss_dict = model._module.loss(x, y=y)
    loss = loss_dict['loss']
    loss.backward()
    mean_per_sample_grad_norm = compute_mean_per_sample_gradient_norm(model, x.shape[0])
    optimizer.step()
    return loss_dict, mean_per_sample_grad_norm


def train(train_loader, val_loader, config, log_dir):
    # reproducibility
    print(f"Setting seed to {config.seed}...")
    seed_everything(config.seed)

    # Init model
    _, model, optimizer = init_model(config)
    # Training
    print('Starting training...')
    step = 0
    i_epoch = 0
    train_losses = AvgDictMeter()
    t_start = time()

    while True:
        model.train()
        for x, y, meta in train_loader:
            step += 1
            # forward
            loss_dict = train_step(model, optimizer, x, y, config.device)
            # add loss
            train_losses.add(loss_dict)
            if step % config.log_frequency == 0:
                train_results = train_losses.compute()
                # Print training loss
                log_msg = " - ".join([f'{k}: {v:.4f}' for k, v in train_results.items()])
                log_msg = f"Iteration {step} - " + log_msg
                # Elapsed time
                elapsed_time = datetime.utcfromtimestamp(time() - t_start)
                log_msg += f" - time: {elapsed_time.strftime('%d-%H:%M:%S')}s"
                # Estimate remaining time
                time_per_step = (time() - t_start) / step
                remaining_steps = config.max_steps - step
                remaining_time = remaining_steps * time_per_step
                remaining_time = datetime.utcfromtimestamp(remaining_time)
                log_msg += f" - remaining time: {remaining_time.strftime('%d-%H:%M:%S')}s"
                print(log_msg)
                # Log to w&b or tensorboard
                wandb.log({f'train/{k}': v for k, v in train_results.items()}, step=step)
                # Reset
                train_losses.reset()

            # validation
            if step % config.val_frequency == 0:
                log_imgs = step % config.log_img_freq == 0
                val_results = validate(config, model, val_loader, step, log_dir, log_imgs)
                # Log to w&b
                wandb.log(val_results, step=step)

            if step >= config.max_steps:
                print(f'Reached {config.max_steps} iterations.', 'Finished training.')
                # Final validation
                print("Final validation...")
                validate(config, model, val_loader, step, log_dir, log_imgs)
                return model
        i_epoch += 1
        print(f'Finished epoch {i_epoch}, ({step} iterations)')


def log_time(remaining_time: float):
    time_left = int(remaining_time)
    time_duration = timedelta(seconds=time_left)
    days = time_duration.days
    hours = time_duration.seconds // 3600
    minutes = (time_duration.seconds // 60) % 60
    seconds = time_duration.seconds % 60
    return f"{days}d-{hours}h-{minutes}m-{seconds}s"

def train_dp(train_loader, val_loader, config, log_dir):
    # Reproducibility
    print(f"Setting seed to {config.seed}...")
    seed_everything(config.seed)

    # Init model
    _, model, optimizer = init_model(config)

    # Init DP
    privacy_engine = PrivacyEngine(accountant="rdp")
    delta = config.delta if config.delta else 1 / (len(train_loader) * train_loader.batch_size)
    epochs = math.ceil(config.max_steps / len(train_loader))
    # to use the complete privacy budget, we need to train for at least as many steps as epochs
    config.max_steps = epochs * len(train_loader)
    # also need to adjust the max_steps to account for the fact that we are using a larger batch size than the GPU
    # can handle
    config.max_steps = (config.batch_size / config.max_physical_batch_size) * config.max_steps
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=config.epsilon,
        target_delta=delta,
        max_grad_norm=config.max_grad_norm,
        epochs=epochs
    )
    # validate model
    errors = ModuleValidator.validate(model, strict=False)
    if len(errors) > 0:
        print(f'WARNING: model validation failed with errors: {errors}')
    # Training
    print('Starting training...')
    step = 0
    i_epoch = 0
    train_losses = AvgDictMeter()
    t_start = time()
    while True:
        with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=config.max_physical_batch_size,
                optimizer=optimizer
        ) as new_train_loader:
            mean_gradient_per_class = {0: 0, 1: 0}
            count_samples_per_class = {0: 0, 1: 0}
            model.train()
            for x, y, meta in new_train_loader:
                step += 1
                # forward
                loss_dict, accumulated_per_sample_norms = train_step_dp(model, optimizer, x, y, config.device)
                # accumulate mean gradient norms per class
                for g_norm, pv_label in zip(torch.squeeze(accumulated_per_sample_norms).tolist(), meta.tolist()):
                    mean_gradient_per_class[pv_label] += g_norm
                    count_samples_per_class[pv_label] += 1
                # add loss
                train_losses.add(loss_dict)

                if step % config.log_frequency == 0:
                    train_results = train_losses.compute()
                    # Print training loss
                    log_msg = " - ".join([f'{k}: {v:.4f}' for k, v in train_results.items()])
                    log_msg = f"Iteration {step} - " + log_msg
                    # Elapsed time
                    elapsed_time = time() - t_start
                    log_msg += f" - time: {log_time(elapsed_time)}"
                    # Estimate remaining time
                    time_per_step = (time() - t_start) / step
                    remaining_steps = config.max_steps - step
                    remaining_time = remaining_steps * time_per_step
                    log_msg += f" - remaining time: {log_time(remaining_time)}"
                    print(log_msg)
                    # Log to w&b or tensorboard
                    wandb.log({f'train/{k}': v for k, v in train_results.items()}, step=step)
                    # Reset
                    train_losses.reset()

                eps = privacy_engine.get_epsilon(delta)
                if step % config.val_frequency == 0:
                    log_imgs = step % config.log_img_freq == 0
                    val_results = validate(config, model, val_loader, step, log_dir, log_imgs)
                    print(f"ɛ: {eps:.2f} (target: 8)")
                    val_results['epsilon'] = eps
                    # Log to w&b
                    wandb.log(val_results, step=step)
                # check if maximum ɛ is reached
                if eps >= config.epsilon:
                    print(f'Reached maximum ɛ {eps}/{config.epsilon}.', 'Finished training.')
                    # Final validation
                    print("Final validation...")
                    validate(config, model, val_loader, step, log_dir, log_imgs)
                    return model

                if step >= config.max_steps:
                    print(f'Reached {config.max_steps} iterations.', 'Finished training.')
                    # Final validation
                    print("Final validation...")
                    validate(config, model, val_loader, step, log_dir, log_imgs)
                    return model
            i_epoch += 1
            mapping = {
                0: "young" if config.protected_attr == 'age' else "male",
                1: "old" if config.protected_attr == 'age' else "female"
            }
            # log mean gradient norms per class to wandb
            wandb.log({"train/mean_grads": {mapping[k]: v/count_samples_per_class[k] if count_samples_per_class[k] != 0 else 0 for k, v in mean_gradient_per_class.items()}}, step=step)
            print(f'Finished epoch {i_epoch}, ({step} iterations)')




""""""""""""""""""""""""""""""""" Validation """""""""""""""""""""""""""""""""


def val_step(model, x, y, meta, device, dp=False):
    model.eval()
    x = x.to(device)
    y = y.to(device)
    meta = meta.to(device)
    with torch.no_grad():
        # TODO: find a better way to compute the loss when wrapped with DP
        if dp:
            loss_dict = model._module.loss(x, y=y)
            anomaly_map, anomaly_score = model._module.predict_anomaly(x)
        else:
            loss_dict = model.loss(x, y=y)
            anomaly_map, anomaly_score = model.predict_anomaly(x)
    x = x.cpu()
    y = y.cpu()
    anomaly_score = anomaly_score.cpu() if anomaly_score is not None else None
    anomaly_map = anomaly_map.cpu() if anomaly_map is not None else None
    return loss_dict, anomaly_map, anomaly_score


def validate(config, model, loader, step, log_dir, log_imgs=False):
    i_step = 0
    device = next(model.parameters()).device
    x, y, meta = next(iter(loader))
    metrics = build_metrics(subgroup_names=list(x.keys()))
    losses = defaultdict(AvgDictMeter)
    imgs = defaultdict(list)
    anomaly_maps = defaultdict(list)

    for x, y, meta in loader:
        # x, y, anomaly_map: [b, 1, h, w]
        # Compute loss, anomaly map and anomaly score
        for i, k in enumerate(x.keys()):
            loss_dict, anomaly_map, anomaly_score = val_step(model, x[k], y[k], meta[k], device, config.dp)

            # Update metrics
            group = torch.tensor([i] * len(anomaly_score))
            metrics.update(group, anomaly_score, y[k])
            losses[k].add(loss_dict)
            imgs[k].append(x[k])
            if anomaly_map is not None:
                anomaly_maps[k].append(anomaly_map)
            else:
                log_imgs = False

        i_step += 1
        if i_step >= config.val_steps:
            break

    # Compute and flatten metrics and losses
    metrics_c = metrics.compute()
    losses_c = {k: v.compute() for k, v in losses.items()}
    losses_c = {f'{k}_{m}': v[m] for k, v in losses_c.items() for m in v.keys()}
    if log_imgs:
        imgs = {f'{k}_imgs': wandb.Image(torch.cat(v)[:config.num_imgs_log]) for k, v in imgs.items()}
        anomaly_maps = {f'{k}_anomaly_maps': wandb.Image(torch.cat(v)[:config.num_imgs_log]) for k, v in
                        anomaly_maps.items()}
        imgs_log = {**imgs, **anomaly_maps}
        wandb.log(imgs_log, step=step)

    # Compute metrics
    results = {**metrics_c, **losses_c}

    # Print validation results
    print("\nval results:")
    log_msg = "\n".join([f'{k}: {v:.4f}' for k, v in results.items()])
    log_msg += "\n"
    print(log_msg)

    # Save checkpoint
    if not config.debug:
        ckpt_name = os.path.join(log_dir, 'ckpt_last.pth')
        print(f'Saving checkpoint to {ckpt_name}')
        save_checkpoint(ckpt_name, model, step, vars(config))
    return results


""""""""""""""""""""""""""""""""" Testing """""""""""""""""""""""""""""""""


def test(config, model, loader, log_dir):
    print("Testing...")

    device = next(model.parameters()).device
    x, y, meta = next(iter(loader))
    metrics = build_metrics(subgroup_names=list(x.keys()))
    losses = defaultdict(AvgDictMeter)
    anomaly_scores = []
    labels = []
    subgroup_names = []

    for x, y, meta in loader:
        # x, y, anomaly_map: [b, 1, h, w]
        # Compute loss, anomaly map and anomaly score
        for i, k in enumerate(x.keys()):
            loss_dict, _, anomaly_score = val_step(model, x[k], y[k], meta[k], device, config.dp)

            # Update metrics
            group = torch.tensor([i] * len(anomaly_score))
            metrics.update(group, anomaly_score, y[k])
            losses[k].add(loss_dict)

            # Store anomaly scores, labels, and subgroup names
            anomaly_scores.append(anomaly_score)
            labels.append(y[k])
            subgroup_names += [k] * len(anomaly_score)

    # Aggregate anomaly scores, labels, and subgroup names
    anomaly_scores = torch.cat(anomaly_scores)
    labels = torch.cat(labels)

    # Compute and flatten metrics and losses
    metrics_c = metrics.compute()
    losses_c = {k: v.compute() for k, v in losses.items()}
    losses_c = {f'{k}_{m}': v[m] for k, v in losses_c.items() for m in v.keys()}

    results = {**metrics_c, **losses_c}

    # Print validation results
    print("\nTest results:")
    log_msg = "\n".join([f'{k}: {v.mean():.4f}' for k, v in results.items()])
    log_msg += "\n"
    print(log_msg)

    # Write test results to wandb summary
    for k, v in results.items():
        wandb.run.summary[k] = v

    # Save test results to csv
    if not config.debug:
        csv_path = os.path.join(log_dir, 'test_results.csv')
        # create dataframe from dict, keys are the columns and values are single row
        metrics_c = {k: v.item() for k, v in metrics_c.items()}
        df = pd.DataFrame.from_dict(metrics_c, orient='index').T
        for k, v in config.items():
            df[k] = pd.Series([v])
        df.to_csv(csv_path, index=False)

    # Save anomaly scores and labels to csv
    if not config.debug:
        # os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, 'anomaly_scores.csv')
        df = pd.DataFrame({'anomaly_score': anomaly_scores, 'label': labels, 'subgroup_name': subgroup_names})
        df.to_csv(csv_path, index=False)


""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""


def hyper_param_sweep(config):
    print("Sweeping...")
    with open('sweep_config.yml', 'r') as f:
        sweep_config = yaml.safe_load(f)
        sweep_configuration = sweep_config["sweep_config"]
        # replace the default config values with the sweep config values
        for arg_n, arg_v in sweep_config["exp_config"].items():
            config[arg_n] = arg_v
        if config.protected_attr == "age":
            config["old_percent"] = config.protected_attr_percent
        else:
            config["male_percent"] = config.protected_attr_percent
        # get log dir
        log_dir, group_name, job_type = construct_log_dir(config, current_time, sweep_configuration)
        sweep_configuration["name"] = group_name
        # load data
        train_loader, val_loader, test_loader = load_data(config)

        execute_functions = lambda func1, func2: (func1(), func2())
        func1 = lambda: init_wandb(config, log_dir, group_name, job_type)
        func2 = lambda: train(train_loader, val_loader, config, log_dir)
        sweep_function = functools.partial(execute_functions, func1, func2)

        sweep_id = wandb.sweep(sweep_configuration, project=config.wandb_project)
        wandb.agent(sweep_id, function=sweep_function, count=config.num_runs)
        wandb.finish()


def run(config, run_config, reverse):
    # update default values config
    # replace the default config values with the run config
    for arg_name, arg_value in run_config.items():
        config[arg_name] = arg_value
    # get protected attribute values (can be a list or a single value)
    protected_attr_values = []
    if isinstance(config.protected_attr_percent, float):
        protected_attr_values = [config.protected_attr_percent]
    elif isinstance(config.protected_attr_percent, list):
        protected_attr_values = list(np.arange(config.protected_attr_percent[0],
                                               config.protected_attr_percent[1] + config.protected_attr_percent[2],
                                               config.protected_attr_percent[2]
                                               ))
    if len(protected_attr_values) > 1 and reverse:
        protected_attr_values = protected_attr_values[::-1]
    # one run per protected attribute value
    for protected_attr_value in protected_attr_values:
        # set the correct protected attribute value
        config.protected_attr_percent = protected_attr_value
        # load data
        train_loader, val_loader, test_loader = load_data(config)
        # iterate over seeds
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
            torch.cuda.empty_cache()
            # sleep to allow cuda cache to be cleared
            time.sleep(3*60)


if __name__ == '__main__':
    # get time
    current_time = datetime.strftime(datetime.now(), format="%Y.%m.%d-%H:%M:%S")
    # copy default config
    new_config = DEFAULT_CONFIG.copy()
    # set initial seed
    new_config.seed = new_config.initial_seed
    # check if we are doing a hyper-param sweep or not
    if RUN_CONFIG.sweep:
        hyper_param_sweep(new_config.copy())
    else:
        # get run configs
        with open('run_config.yml', 'r') as f:
            run_configs = yaml.safe_load(f)
        # choose runs to execute or run all
        run_configs = run_configs if RUN_CONFIG.run_all else {RUN_CONFIG.run_name: run_configs[RUN_CONFIG.run_name]}
        for run_name, r_config in run_configs.items():
            print(f"Running {run_name}")
            run(new_config.copy(), r_config, RUN_CONFIG.reverse)
