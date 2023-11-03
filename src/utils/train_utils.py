import torch
import wandb
import os
from opacus.utils.batch_memory_manager import BatchMemoryManager
from collections import defaultdict
import pandas as pd
from src.models.DeepSVDD.deepsvdd import DeepSVDD
from src.models.FAE.fae import FeatureReconstructor
from src.models.RD.reverse_distillation import ReverseDistillation
from src.utils.metrics import AvgDictMeter, build_metrics
from src.utils.utils import log_time, get_subgroup_loss_weights
from opacus.validators import ModuleValidator
from time import time
from datetime import datetime
from src.data.datasets import get_dataloaders_rsna, get_dataloaders_other
from dotmap import DotMap
from opacus.validators.utils import register_module_fixer
from torch import nn

@register_module_fixer([nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm])
def _batchnorm_to_groupnorm(module) -> nn.GroupNorm:
    num_groups = 32
    if module.num_features % num_groups != 0:
        num_groups = 25
    return nn.GroupNorm(min(num_groups, module.num_features), module.num_features, affine=module.affine)


DEFAULT_CONFIG = {
    # General script settings
    "initial_seed": 1,
    "num_seeds": 1,
    "debug": False,
    "disable_wandb": False,
    "wandb_project": "unsupervised-fairness",
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
    "no_img_log": False,
    "log_dir": os.path.join('logs', datetime.strftime(datetime.now(), format="%Y.%m.%d-%H:%M:%S")),
    # Hyperparameters
    "lr": 2e-4,
    "weight_decay": 0.0,
    "num_steps": 45000,
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
    "max_grad_norm": 0.01,
    # Other
    "group_name_mod": None,
    "job_type_mod": None,
    "max_physical_batch_size": 512,
    "loss_weight_type": None,
    "weight": 1,
    "n_adam": False,
    "wb_custom_run_name": None,
    "second_stage_epsilon": None,
    "second_stage_steps": None,
    "upsampling_strategy": None,
    "custom_sr": False,
    "effective_dataset_size": 1.0,
    "dataset_random_state": 42,
    "n_training_samples": None,
    "best_and_worst_subsets":None
}
DEFAULT_CONFIG = DotMap(DEFAULT_CONFIG)
DEFAULT_CONFIG.device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {DEFAULT_CONFIG.device}")


def load_data(config):
    print("Loading data...")
    t_load_data_start = time()
    if config.dataset == "rsna-old":
        train_loaders, val_loader, test_loader, max_sample_freq = get_dataloaders_rsna(
            dataset=config.dataset,
            batch_size=config.batch_size,
            img_size=config.img_size, num_workers=config.num_workers,
            protected_attr=config.protected_attr,
            male_percent=config.protected_attr_percent,
            old_percent=config.protected_attr_percent,
            upsampling_strategy=config.upsampling_strategy,
            effective_dataset_size=config.effective_dataset_size,
            random_state = config.dataset_random_state,
            n_training_samples=config.n_training_samples,
            best_and_worst_subsets=config.best_and_worst_subsets
        )
    else:
        train_loaders, val_loader, test_loader = get_dataloaders_other(
            dataset=config.dataset,
            batch_size=config.batch_size,
            img_size=config.img_size,
            num_workers=config.num_workers,
            protected_attr=config.protected_attr,
            n_training_samples=config.n_training_samples,
            use_best_samples=config.use_best_samples
        )
        max_sample_freq = 1

    print(f'Loaded datasets in {time() - t_load_data_start:.2f}s')
    if config.n_training_samples or config.best_and_worst_subsets or config.use_best_samples != "":
        return train_loaders, val_loader, test_loader, max_sample_freq
    else:
        return train_loaders[0], val_loader, test_loader, max_sample_freq


def init_model(config):
    print("Initializing model...")
    if config.model_type == 'FAE':
        model = FeatureReconstructor(config)
    elif config.model_type == 'DeepSVDD':
        model = DeepSVDD(config)
    elif config.model_type == 'RD':
        model = ReverseDistillation(config)
    else:
        raise ValueError(f'Unknown model type {config.model_type}')
    model = model.to(config.device)

    # perform model surgery if DP is enabled
    if config.dp:
        model = ModuleValidator.fix(model)

    # Init optimizer
    if config.n_adam:
        optimizer = torch.optim.NAdam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    return model, optimizer


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
    # print something
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


def train_step_dp(model, optimizer, x, y, loss_weights):
    model.train()
    optimizer.zero_grad()
    loss_dict = model._module.loss(x, per_sample_loss_weights=loss_weights)
    loss = loss_dict['loss']
    loss.backward()
    mean_per_sample_grad_norm = compute_mean_per_sample_gradient_norm(model, x.shape[0])
    optimizer.step()
    return loss_dict, mean_per_sample_grad_norm


def train(model, optimizer, train_loader, val_loader, config, log_dir, prev_step=0):
    # Training
    print('Starting training...')
    i_step = prev_step
    i_epoch = 0
    train_losses = AvgDictMeter()
    t_start = time()
    loss_weights = get_subgroup_loss_weights((config.protected_attr_percent, 1 - config.protected_attr_percent), config)
    log_imgs = False
    while True:
        model.train()
        for x, y, meta in train_loader:
            i_step += 1
            # forward
            per_sample_loss_weights = torch.where(meta == 0, loss_weights[0], loss_weights[1]).to(config["device"])
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
                log_imgs = (i_step % config.log_img_freq == 0) and not config.no_img_log
                val_results = validate(config, model, optimizer, val_loader, i_step, log_dir, log_imgs)
                # Log to w&b
                wandb.log(val_results, step=i_step)

        i_epoch += 1
        if i_epoch % 100 == 0:
            print(f'Finished epoch {i_epoch}/{config.epochs}, ({i_step} iterations)')
        if i_epoch >= config.epochs:
            print(f'Reached {config.epochs} epochs.', 'Finished training.')
            # Final validation
            print("Final validation...")
            validate(config, model, optimizer, val_loader, i_step, log_dir, log_imgs)
            return model, i_step


def train_dp(model, optimizer, train_loader, val_loader, config, log_dir, privacy_engine, prev_step=0):
    # validate model
    errors = ModuleValidator.validate(model, strict=False)
    if len(errors) > 0:
        print(f'WARNING: model validation failed with errors: {errors}')
    # Training
    print('Starting training...')
    i_step = prev_step
    i_epoch = 0
    train_losses = AvgDictMeter()
    t_start = time()
    loss_weights = get_subgroup_loss_weights((config.protected_attr_percent, 1-config.protected_attr_percent), config)
    log_imgs = False
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
                x = x.to(config["device"])
                y = y.to(config["device"])
                i_step += 1
                # compute weights loss weights
                per_sample_loss_weights = torch.where(meta == 0, loss_weights[0], loss_weights[1]).to(config["device"])
                # forward
                loss_dict, accumulated_per_sample_norms = train_step_dp(model, optimizer, x, y, per_sample_loss_weights)
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

                eps = privacy_engine.get_epsilon(config.delta)
                if i_step % config.val_frequency == 0:
                    log_imgs = i_step % config.log_img_freq == 0 and not config.no_img_log
                    val_results = validate(config, model, optimizer, val_loader, i_step, log_dir, log_imgs, privacy_engine=privacy_engine)
                    print(f"ɛ: {eps:.2f} (target: {config.epsilon})")
                    val_results['epsilon'] = eps
                    # Log to w&b
                    wandb.log(val_results, step=i_step)
                # check if maximum ɛ is reached
                if eps >= config.epsilon:
                    print(f'Reached maximum ɛ {eps}/{config.epsilon}.', 'Finished training.')
                    # Final validation
                    print("Final validation...")
                    validate(config, model, optimizer, val_loader, i_step, log_dir, log_imgs, privacy_engine=privacy_engine)
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
                print(f"ɛ: {eps:.2f} (target: {config.epsilon})")
                # Final validation
                print("Final validation...")
                validate(config, model, optimizer, val_loader, i_step, log_dir, log_imgs, privacy_engine=privacy_engine)
                return model, i_step


""""""""""""""""""""""""""""""""" Validation """""""""""""""""""""""""""""""""


def val_step(model, x, y, meta, device, dp=False):
    model.eval()
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
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


def validate(config, model, optimizer, loader, step, log_dir, log_imgs=False, privacy_engine=None):
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
        pass
        #ckpt_name = os.path.join(log_dir, 'ckpt_last.pth')
        #print(f'Saving checkpoint to {ckpt_name}')
        #if config.dp:
        #    save_checkpoint_dp(ckpt_name, model, optimizer, privacy_engine, step, dict(config))
        #else:
        #    save_checkpoint(ckpt_name, model, optimizer, step, dict(config))
    return results


""""""""""""""""""""""""""""""""" Testing """""""""""""""""""""""""""""""""


def test(config, model, loader, log_dir, stage_two=False):
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
        csv_path = os.path.join(log_dir, 'test_results.csv' if not stage_two else 'test_results_stage_two.csv')
        # create dataframe from dict, keys are the columns and values are single row
        metrics_c = {k: v.item() for k, v in metrics_c.items()}
        df = pd.DataFrame.from_dict(metrics_c, orient='index').T
        for k, v in config.items():
            df[k] = pd.Series([v])
        df.to_csv(csv_path, index=False)

    # Save anomaly scores and labels to csv
    if not config.debug:
        # os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, 'anomaly_scores.csv' if not stage_two else 'anomaly_scores_stage_two.csv')
        df = pd.DataFrame({'anomaly_score': anomaly_scores, 'label': labels, 'subgroup_name': subgroup_names})
        df.to_csv(csv_path, index=False)
