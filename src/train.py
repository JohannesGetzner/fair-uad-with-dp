import functools
import sys

import numpy as np
import yaml

from opacus.utils.batch_memory_manager import BatchMemoryManager

sys.path.append('..')
import os
from argparse import ArgumentParser
from collections import defaultdict

from dotmap import DotMap
from time import time
from datetime import datetime

import pandas as pd
import torch
import wandb

from src.data.datasets import get_dataloaders
from src.models.DeepSVDD.deepsvdd import DeepSVDD
from src.models.FAE.fae import FeatureReconstructor
from src.utils.metrics import AvgDictMeter, build_metrics
from src.utils.utils import seed_everything, save_checkpoint, init_wandb, construct_log_dir, log_time, get_subgroup_loss_weights
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
    "epochs": 100,
    # "max_steps": 8000,
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
    "weigh_loss": None,
}
DEFAULT_CONFIG = DotMap(DEFAULT_CONFIG)

parser = ArgumentParser()
parser.add_argument('--run_name', default="dp_1", type=str)
parser.add_argument('--job_type_mod', default="", type=str)
parser.add_argument('--protected_attr_percent', default=0.5, type=float)
parser.add_argument('--custom', default=0.9, type=float)
print(str(datetime.strftime(datetime.now(), format="%Y.%m.%d-%H:%M:%S")))
parser.add_argument('--d', type=str, default=str(datetime.strftime(datetime.now(), format="%Y.%m.%d-%H:%M:%S")))
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


def train_step(model, optimizer, x, y, device, loss_weights):
    model.train()
    optimizer.zero_grad()
    x = x.to(device)
    y = y.to(device)
    loss_weights = loss_weights.to(device)
    loss_dict = model.loss(x, per_sample_loss_weights=loss_weights)
    loss = loss_dict['loss']
    loss.backward()
    optimizer.step()
    return loss_dict


def train_step_dp(model, optimizer, x, y, device, loss_weights):
    model.train()
    optimizer.zero_grad()
    x = x.to(device)
    y = y.to(device)
    loss_weights = loss_weights.to(device)
    loss_dict = model._module.loss(x, per_sample_loss_weights=loss_weights)
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
    i_step = 0
    i_epoch = 0
    train_losses = AvgDictMeter()
    t_start = time()
    loss_weights = get_subgroup_loss_weights(
        (config.protected_attr_percent, 1 - config.protected_attr_percent),
        mode=config.weigh_loss,
        dp=False
    )
    while True:
        model.train()
        for x, y, meta in train_loader:
            i_step += 1
            # forward
            per_sample_loss_weights = torch.where(meta == 0, loss_weights[0], loss_weights[1])
            loss_dict = train_step(model, optimizer, x, y, config.device, per_sample_loss_weights)
            # add loss
            train_losses.add(loss_dict)
            if i_step % config.log_frequency == 0:
                train_results = train_losses.compute()
                # Print training loss
                log_msg = " - ".join([f'{k}: {v:.4f}' for k, v in train_results.items()])
                log_msg = f"Iteration {i_step} - " + log_msg
                # Elapsed time
                elapsed_time = datetime.utcfromtimestamp(time() - t_start)
                log_msg += f" - time: {elapsed_time.strftime('%d-%H:%M:%S')}s"
                # Estimate remaining time
                time_per_epoch = ((time() - t_start) / i_epoch) if i_epoch > 0 else time() - t_start
                remaining_time = (config.epochs - i_epoch) * time_per_epoch
                log_msg += f" - remaining time: {log_time(remaining_time)}"
                print(log_msg)
                # Log to w&b or tensorboard
                wandb.log({f'train/{k}': v for k, v in train_results.items()}, step=i_step)
                # Reset
                train_losses.reset()

            # validation
            if i_step % config.val_frequency == 0:
                log_imgs = i_step % config.log_img_freq == 0
                val_results = validate(config, model, val_loader, i_step, log_dir, log_imgs)
                # Log to w&b
                wandb.log(val_results, step=i_step)

        i_epoch += 1
        print(f'Finished epoch {i_epoch}/{config.epochs}, ({i_step} iterations)')
        if i_epoch >= config.epochs:
            print(f'Reached {config.epochs} epochs.', 'Finished training.')
            # Final validation
            print("Final validation...")
            validate(config, model, val_loader, i_step, log_dir, log_imgs)
            return model


def train_dp(train_loader, val_loader, config, log_dir):
    # Reproducibility
    print(f"Setting seed to {config.seed}...")
    seed_everything(config.seed)

    # Init model
    _, model, optimizer = init_model(config)

    # Init DP
    privacy_engine = PrivacyEngine(accountant="rdp")
    delta = config.delta if config.delta else 1 / (len(train_loader) * train_loader.batch_size)
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=config.epsilon,
        target_delta=delta,
        max_grad_norm=config.max_grad_norm,
        epochs=config.epochs
    )
    # validate model
    errors = ModuleValidator.validate(model, strict=False)
    if len(errors) > 0:
        print(f'WARNING: model validation failed with errors: {errors}')
    # Training
    print('Starting training...')
    i_step = 0
    i_epoch = 0
    train_losses = AvgDictMeter()
    t_start = time()
    loss_weights = get_subgroup_loss_weights(
        (config.protected_attr_percent, 1-config.protected_attr_percent),
        mode = config.weigh_loss,
        dp=True
    )
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
                i_step += 1
                # compute weights loss weights
                per_sample_loss_weights = torch.where(meta == 0, loss_weights[0], loss_weights[1])
                # forward
                loss_dict, accumulated_per_sample_norms = train_step_dp(model, optimizer, x, y, config.device, per_sample_loss_weights)
                # accumulate mean gradient norms per class
                for g_norm, pv_label in zip(torch.squeeze(accumulated_per_sample_norms).tolist(), meta.tolist()):
                    mean_gradient_per_class[pv_label] += g_norm
                    count_samples_per_class[pv_label] += 1
                # add loss
                train_losses.add(loss_dict)

                if i_step % config.log_frequency == 0:
                    train_results = train_losses.compute()
                    # Print training loss
                    log_msg = " - ".join([f'{k}: {v:.4f}' for k, v in train_results.items()])
                    log_msg = f"Iteration {i_step} - " + log_msg
                    # Elapsed time
                    elapsed_time = time() - t_start
                    log_msg += f" - time: {log_time(elapsed_time)}"
                    # Estimate remaining time
                    time_per_epoch = ((time() - t_start) / i_epoch) if i_epoch > 0 else time() - t_start
                    remaining_time = (config.epochs - i_epoch) * time_per_epoch
                    log_msg += f" - remaining time: {log_time(remaining_time)}"
                    print(log_msg)
                    # Log to w&b or tensorboard
                    wandb.log({f'train/{k}': v for k, v in train_results.items()}, step=i_step)
                    # Reset
                    train_losses.reset()

                eps = privacy_engine.get_epsilon(delta)
                if i_step % config.val_frequency == 0:
                    log_imgs = i_step % config.log_img_freq == 0
                    val_results = validate(config, model, val_loader, i_step, log_dir, log_imgs)
                    print(f"ɛ: {eps:.2f} (target: 8)")
                    val_results['epsilon'] = eps
                    # Log to w&b
                    wandb.log(val_results, step=i_step)
                # check if maximum ɛ is reached
                if eps >= config.epsilon:
                    print(f'Reached maximum ɛ {eps}/{config.epsilon}.', 'Finished training.')
                    # Final validation
                    print("Final validation...")
                    validate(config, model, val_loader, i_step, log_dir, log_imgs)
                    return model
            i_epoch += 1
            print(f'Finished epoch {i_epoch}/{config.epochs}, ({i_step} iterations)')

            # log mean gradient norms per class to wandb
            mapping = {
                1: "young" if config.protected_attr == 'age' else "female",
                0: "old" if config.protected_attr == 'age' else "male"}
            wandb.log({"train/mean_grads": {
                mapping[k]: v / count_samples_per_class[k] if count_samples_per_class[k] != 0 else 0 for k, v in
                mean_gradient_per_class.items()}}, step=i_step)

            if i_epoch >= config.epochs:
                print(f'Reached {config.epochs} epochs.', 'Finished training.')
                # Final validation
                print("Final validation...")
                validate(config, model, val_loader, i_step, log_dir, log_imgs)
                return model


""""""""""""""""""""""""""""""""" Validation """""""""""""""""""""""""""""""""


def val_step(model, x, y, meta, device, dp=False):
    model.eval()
    x = x.to(device)
    y = y.to(device)
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
