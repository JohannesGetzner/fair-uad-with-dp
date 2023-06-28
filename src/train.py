import functools
import sys

import numpy as np
import yaml
from opacus.utils.batch_memory_manager import BatchMemoryManager

sys.path.append('..')
import os
from argparse import ArgumentParser, BooleanOptionalAction
from collections import defaultdict
from datetime import datetime
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
from src.utils.utils import seed_everything, save_checkpoint, init_wandb
from opacus import PrivacyEngine
from opacus.validators.utils import register_module_fixer
from torch import nn
from opacus.validators import ModuleValidator
from src.models.pytorch_ssim import SSIMLoss


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
    "wandb_project": "test",
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
    "max_grad_norm": 1.0
}
DEFAULT_CONFIG = DotMap(DEFAULT_CONFIG)

parser = ArgumentParser()
parser.add_argument('--sweep', action=BooleanOptionalAction, default=False)
parser.add_argument('--run_all', action=BooleanOptionalAction, default=False)
parser.add_argument('--run_name', default="run1", type=str)
RUN_SETTINGS = parser.parse_args()

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


def train_step(model, optimizer, x, y, meta, device, dp=False):
    model.train()
    optimizer.zero_grad()
    x = x.to(device)
    y = y.to(device)
    meta = meta.to(device)
    # TODO: find a better way to compute the loss when wrapped with DP (don't access protected attribute)
    if dp:
        loss_dict = model._module.loss(x, y=y)
    else:
        loss_dict = model.loss(x, y=y)
    loss = loss_dict['loss']
    loss.backward()
    optimizer.step()
    return loss_dict


def train(train_loader, val_loader, config, log_dir):
    # Reproducibility
    print(f"Setting seed to {config.seed}...")
    seed_everything(config.seed)

    # Init model
    _, model, optimizer = init_model(config)
    loss_fn = SSIMLoss(window_size=5, size_average=False)

    # Init logging
    if not config.debug:
        os.makedirs(log_dir, exist_ok=True)
    run = init_wandb(config, config.wandb_project, log_dir)
    # Init DP
    if config.dp:
        privacy_engine = PrivacyEngine(accountant="rdp")
        epochs = math.ceil(config.max_steps / len(train_loader))
        delta = config.delta if config.delta else 1 / (len(train_loader) * train_loader.batch_size)
        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=config.epsilon,
            target_delta=delta,
            max_grad_norm=config.max_grad_norm if not config.sweep else wandb.config.max_grad_norm,
            epochs=epochs
        )
        errors = ModuleValidator.validate(model, strict=False)
        if len(errors) > 0:
            print(f'WARNING: model validation failed with errors: {errors}')
    print('Starting training...')
    step = 0
    i_epoch = 0
    train_losses = AvgDictMeter()
    t_start = time()
    while True:
        with BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=512, optimizer=optimizer) \
                as new_train_loader:
            for x, y, meta in new_train_loader:
                step += 1

                loss_dict = train_step(model, optimizer, x, y, meta, config.device, config.dp)
                train_losses.add(loss_dict)

                if step % config.log_frequency == 0:
                    train_results = train_losses.compute()
                    # Print training loss
                    log_msg = " - ".join([f'{k}: {v:.4f}' for k, v in train_results.items()])
                    log_msg = f"Iteration {step} - " + log_msg
                    # Elapsed time
                    elapsed_time = datetime.utcfromtimestamp(time() - t_start)
                    log_msg += f" - time: {elapsed_time.strftime('%H:%M:%S')}s"
                    # Estimate remaining time
                    time_per_step = (time() - t_start) / step
                    remaining_steps = config.max_steps - step
                    remaining_time = remaining_steps * time_per_step
                    remaining_time = datetime.utcfromtimestamp(remaining_time)
                    log_msg += f" - remaining time: {remaining_time.strftime('%H:%M:%S')}"
                    print(log_msg)

                    # Log to w&b or tensorboard
                    wandb.log({f'train/{k}': v for k, v in train_results.items()}, step=step)

                    # Reset
                    train_losses.reset()

                if step % config.val_frequency == 0:
                    log_imgs = step % config.log_img_freq == 0
                    val_results = validate(config, model, val_loader, step, log_dir, log_imgs)
                    if config.dp:
                        eps = privacy_engine.get_epsilon(delta)
                        print(f"ɛ: {eps:.2f} (target: 8)")
                        val_results['epsilon'] = eps
                    # Log to w&b
                    wandb.log(val_results, step=step)
                    if config.dp:
                        if eps > config.epsilon:
                            print(f'Reached maximum ɛ {eps}/{config.epsilon}.', 'Finished training.')
                            # Final validation
                            print("Final validation...")
                            validate(config, model, val_loader, step, log_dir, False)
                            return model

                if step >= config.max_steps:
                    print(f'Reached {config.max_steps} iterations.', 'Finished training.')

                    # Final validation
                    print("Final validation...")
                    validate(config, model, val_loader, step, log_dir, False)
                    return model

            i_epoch += 1
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
        for k, v in vars(config).items():
            df[k] = pd.Series([v])
        df.to_csv(csv_path, index=False)

    # Save anomaly scores and labels to csv
    if not config.debug:
        # os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, 'anomaly_scores.csv')
        df = pd.DataFrame({'anomaly_score': anomaly_scores, 'label': labels, 'subgroup_name': subgroup_names})
        df.to_csv(csv_path, index=False)


""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""

if __name__ == '__main__':
    current_time = datetime.strftime(datetime.now(), format="%Y.%m.%d-%H:%M:%S")
    DEFAULT_CONFIG.sweep = RUN_SETTINGS.sweep
    if RUN_SETTINGS.sweep:
        print("Sweeping...")
        config = DEFAULT_CONFIG.copy()
        config.seed = config.initial_seed
        if config.dp:
            config.log_dir += '_DP'
        with open('sweep_config.yml', 'r') as f:
            sweep_configs = yaml.safe_load(f)
        sweep_configs = sweep_configs if RUN_SETTINGS.run_all else {RUN_SETTINGS.run_name: sweep_configs[RUN_SETTINGS.run_name]}
        for sweep_name, sweep_config in sweep_configs.items():
            sweep_configuration = sweep_config["sweep_config"]
            for arg_name, arg_value in sweep_config["exp_config"].items():
                config[arg_name] = arg_value
            config.experiment_name += f"-{current_time}"
            sweep_configuration["name"] = config.experiment_name
            if config.protected_attr == "age":
                config["old_percent"] = config.protected_attr_percent
            else:
                config["male_percent"] = config.protected_attr_percent
            sweep_id = wandb.sweep(sweep_configuration, project=config.wandb.wandb_project)
            train_loader, val_loader, test_loader = load_data(config)
            train_p = functools.partial(train, train_loader, val_loader, config, config.log_dir)
            wandb.agent(sweep_id, function=train_p, count=config.num_runs)
            wandb.finish()
    else:
        with open('run_config.yml', 'r') as f:
            run_configs = yaml.safe_load(f)
        # choose runs to execute
        run_configs = run_configs if RUN_SETTINGS.run_all else {RUN_SETTINGS.run_name: run_configs[RUN_SETTINGS.run_name]}
        for run_name, run_config in run_configs.items():
            print(f"Running {run_name}")
            # update default values config
            config = DEFAULT_CONFIG.copy()
            if config.dp:
                config.log_dir += '_DP'
            protected_attr_values = []
            # set protected attribute values
            for arg_name, arg_value in run_config.items():
                if arg_name == "protected_attr_percent":
                    if isinstance(arg_value, float):
                        protected_attr_values.append(arg_value)
                    else:
                        protected_attr_values += list(np.arange(arg_value[0], arg_value[1]+arg_value[2], arg_value[2]))
                else:
                    config[arg_name] = arg_value
            # add timestamp and protected attribute to experiment name
            config.experiment_name += f"-{current_time}"
            if config.dp:
                config.log_dir += '_DP'
            for protected_attr_value in protected_attr_values:
                log_dir = config.log_dir+ f"_{str(protected_attr_value)[:4].replace('.', '')}"
                if config.protected_attr == "age":
                    config["old_percent"] = protected_attr_value
                else:
                    config["male_percent"] = protected_attr_value
                # load data
                train_loader, val_loader, test_loader = load_data(config)
                # iterate over seeds
                log_dir_no_seed = log_dir
                for i in range(config.num_seeds):
                    config.seed = config.initial_seed + i
                    log_dir = os.path.join(log_dir_no_seed, f'seed_{config.seed}')
                    final_model = train(train_loader, val_loader, config, log_dir)
                    test(config, final_model, test_loader, log_dir)
                    wandb.finish()
