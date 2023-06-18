import sys
sys.path.append('..')
import os
from argparse import ArgumentParser,BooleanOptionalAction
from collections import defaultdict
from datetime import datetime
from time import time

import pandas as pd
import torch
import wandb
import math

from src.data.datasets import get_dataloaders
from src.models.DeepSVDD.deepsvdd import DeepSVDD
from src.models.FAE.fae import FeatureReconstructor
from src.utils.metrics import AvgDictMeter, build_metrics
from src.utils.utils import seed_everything, save_checkpoint
from opacus import PrivacyEngine
from opacus.validators.utils import register_module_fixer
from torch import nn
from opacus.validators import ModuleValidator
from src.models.pytorch_ssim import SSIMLoss


@register_module_fixer(
    [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm]
)
def _batchnorm_to_groupnorm(module) -> nn.GroupNorm:
    num_groups = 32
    if module.num_features % num_groups != 0:
        num_groups = 25
    return nn.GroupNorm(
        min(num_groups, module.num_features), module.num_features, affine=module.affine
    )


""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""

parser = ArgumentParser()
# General script settings
parser.add_argument('--initial_seed', type=int, default=1, help='Random seed')
parser.add_argument('--num_seeds', type=int, default=1, help='Number of random seed')
parser.add_argument('--debug', action='store_true', help='Debug mode')
parser.add_argument('--disable_wandb', action='store_true', help='Debug mode')

# Experiment settings
parser.add_argument('--experiment_name', type=str, default='')

# Data settings
parser.add_argument('--dataset', type=str, default='rsna', choices=['rsna'])
parser.add_argument('--protected_attr', type=str, default='none', choices=['none', 'age', 'sex'])
parser.add_argument('--male_percent', type=float, default=0.5)
parser.add_argument('--old_percent', type=float, default=0.5)
parser.add_argument('--img_size', type=int, default=128, help='Image size')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')

# Logging settings
parser.add_argument('--val_frequency', type=int, default=200, help='Validation frequency')
parser.add_argument('--val_steps', type=int, default=50, help='Steps per validation')
parser.add_argument('--log_frequency', type=int, default=100, help='Logging frequency')
parser.add_argument('--log_img_freq', type=int, default=1000)
parser.add_argument('--num_imgs_log', type=int, default=8)
parser.add_argument('--log_dir', type=str, help="Logging directory",
    default=os.path.join('logs', datetime.strftime(datetime.now(), format="%Y.%m.%d-%H:%M:%S")))

# Hyperparameters
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
parser.add_argument('--max_steps', type=int, default=8000,  # 8000,  # 10000,
                    help='Number of training steps')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

# Model settings
parser.add_argument('--model_type', type=str, default='FAE', choices=['FAE', 'DeepSVDD'])
# FAE settings
# 128, 160, 224, 320
parser.add_argument('--hidden_dims', type=int, nargs='+', default=[100, 150, 200, 300],
                    help='Autoencoder hidden dimensions')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--loss_fn', type=str, default='ssim', help='loss function', choices=['mse', 'ssim'])
parser.add_argument('--extractor_cnn_layers', type=str, nargs='+', default=['layer0', 'layer1', 'layer2'])
parser.add_argument('--keep_feature_prop', type=float, default=1.0, help='Proportion of ResNet features to keep')
# DeepSVDD settings
parser.add_argument('--repr_dim', type=int, default=256, help='Dimensionality of the hypersphere c')

# Differential Privacy settings
parser.add_argument('--dp', action=BooleanOptionalAction, default=False, help='Use differential privacy')
parser.add_argument('--e', type=float, default=8, help='Noise multiplier')
parser.add_argument('--delta', type=float, help='Target delta')
parser.add_argument('--max_grad_norm', type=float, default=1, help='Max gradient norm')

config = parser.parse_args()

config.seed = config.initial_seed
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {config.device}")

if config.debug:
    config.num_workers = 0
    config.max_steps = 1
    config.val_frequency = 1
    config.val_steps = 1
    config.log_frequency = 1

""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""

print("Loading data...")
t_load_data_start = time()
train_loader, val_loader, test_loader = get_dataloaders(dataset=config.dataset, batch_size=config.batch_size,
    img_size=config.img_size, num_workers=config.num_workers, protected_attr=config.protected_attr,
    male_percent=config.male_percent, old_percent=config.old_percent, )
print(f'Loaded datasets in {time() - t_load_data_start:.2f}s')

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

    compiled_model = model #torch.compile(model)

    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    return model, compiled_model, optimizer


""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""


def train_step(model, optimizer, x, y, meta, device):
    model.train()
    optimizer.zero_grad()
    x = x.to(device)
    y = y.to(device)
    meta = meta.to(device)
    # TODO: find a better way to compute the loss when wrapped with DP (don't access protected attribute)
    if config.dp:
        loss_dict = model._module.loss(x, y=y)
    else:
        loss_dict = model.loss(x, y=y)
    loss = loss_dict['loss']
    loss.backward()
    optimizer.step()
    return loss_dict


def train(train_loader, val_loader, config, log_dir, experiment_date):
    # Reproducibility
    print(f"Setting seed to {config.seed}...")
    seed_everything(config.seed)

    # Init model
    _, model, optimizer = init_model(config)
    loss_fn = SSIMLoss(window_size=5, size_average=False)

    # Init logging
    if not config.debug:
        os.makedirs(log_dir, exist_ok=True)
    wandb_tags = [config.model_type, config.dataset, config.protected_attr]
    wandb_group_name = log_dir[:log_dir.rindex('/')]
    wandb_group_name += f"_{experiment_date}"
    wandb.init(
        project="unsupervised-fairness",
        config=config,
        group=wandb_group_name,
        dir=log_dir,
        tags=wandb_tags,
        job_type="seed_"+str(config.seed),
        mode="disabled" if (config.debug or config.disable_wandb) else "online"
    )
    # Init DP
    if config.dp:
        privacy_engine = PrivacyEngine(accountant="rdp")
        epochs = math.ceil(config.max_steps / len(train_loader))
        delta = config.delta if config.delta else 1/(len(train_loader)*train_loader.batch_size)
        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=config.e,
            target_delta=delta,
            max_grad_norm=config.max_grad_norm,
            epochs=epochs,
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
        for x, y, meta in train_loader:
            step += 1

            loss_dict = train_step(model, optimizer, x, y, meta, config.device)
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
                val_results = validate(config, model, val_loader, step, log_imgs)
                # Log to w&b
                wandb.log(val_results, step=step)

            if step >= config.max_steps:
                print(f'Reached {config.max_steps} iterations.', 'Finished training.')

                # Final validation
                print("Final validation...")
                validate(config, model, val_loader, step, False)
                return model

        i_epoch += 1
        print(f'Finished epoch {i_epoch}, ({step} iterations)')


""""""""""""""""""""""""""""""""" Validation """""""""""""""""""""""""""""""""


def val_step(model, x, y, meta, device):
    model.eval()
    x = x.to(device)
    y = y.to(device)
    meta = meta.to(device)
    with torch.no_grad():
        # TODO: find a better way to compute the loss when wrapped with DP
        if config.dp:
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


def validate(config, model, loader, step, log_imgs=False):
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
            loss_dict, anomaly_map, anomaly_score = val_step(model, x[k], y[k], meta[k], device)

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
            loss_dict, _, anomaly_score = val_step(model, x[k], y[k], meta[k], device)

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
    experiment_date = datetime.strftime(datetime.now(), format='%Y.%m.%d-%H:%M:%S')
    for i in range(config.num_seeds):
        torch.cuda.empty_cache()
        config.seed = config.initial_seed + i
        log_dir = config.log_dir
        if config.dp:
            log_dir += '_DP'
        log_dir = os.path.join(log_dir, f'seed_{config.seed}')
        #print(log_dir)
        model = train(train_loader, val_loader, config, log_dir, experiment_date)
        test(config, model, test_loader, log_dir)
        wandb.finish()
