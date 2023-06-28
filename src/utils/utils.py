import functools
import os
import random
from argparse import Namespace
from numbers import Number
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.tensorboard import SummaryWriter


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def save_checkpoint(path: str, model: nn.Module, step: int, config: Dict):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    checkpoint = {
        'model': model.state_dict(),
        'config': config,
        'step': step
    }
    torch.save(checkpoint, path)


def init_wandb(config, log_dir: str, group_name: str, job_type: str):
    sweep_tag = "sweep" if config.sweep_param else "no_sweep"
    dp_tag = "DP" if config.dp else "no_DP"
    wandb_tags = [config.model_type, config.dataset, config.protected_attr, dp_tag, sweep_tag]
    if config.sweep_param:
        run = wandb.init(
            # project=config.wandb_project,
            config=config,
            # dir=log_dir,
            tags=wandb_tags,
            job_type=job_type,
            group=group_name,
            mode="disabled" if (config.debug or config.disable_wandb) else "online"
        )
    else:
        run = wandb.init(
            project=config.wandb_project,
            config=config,
            group=group_name,
            dir=log_dir,
            tags=wandb_tags,
            job_type=job_type,
            name="seed_" + str(config.seed),
            mode="disabled" if (config.debug or config.disable_wandb) else "online"
        )
    return run


def construct_log_dir(config, current_time, sweep_config=None):
    log_path = ""
    # job_type is always the same
    if config.protected_attr == "sex":
        jt = f"male_percent"
    else:
        jt = "old_percent"
    jt += f"_{str(config.protected_attr_percent).replace('.', '')}"
    if config.dp:
        jt += "_DP"
    # build log_dir and group_name
    if config.sweep_param:
        min_val = sweep_config["parameters"][config.sweep_param]["min"]
        max_val = sweep_config["parameters"][config.sweep_param]["max"]
        log_path += f"{config.model_type}-{config.dataset}-{config.sweep_param}-{str(min_val).replace('.','')}-{str(max_val).replace('.','')}-{current_time}"
        gn = "sweep_"+log_path
        log_path = f"{log_path}/{jt}"
    else:
        log_path += f"{config.model_type}-{config.dataset}-{config.protected_attr}-{current_time}"
        gn = log_path
        log_path = f"{log_path}/{jt}/seed_{config.seed}"
    # prepend dirs to log_path
    log_path = f"logs/{'sweeps/' if config.sweep_param else ''}{log_path}"
    return log_path, gn, jt



class TensorboardLogger(SummaryWriter):
    def __init__(
        self,
        log_dir: str = None,
        config: Namespace = None,
        enabled: bool = True,
        comment: str = '',
        purge_step: int = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = ''
    ):
        self.enabled = enabled
        if self.enabled:
            super().__init__(
                log_dir=log_dir,
                comment=comment,
                purge_step=purge_step,
                max_queue=max_queue,
                flush_secs=flush_secs,
                filename_suffix=filename_suffix
            )
        else:
            return

        # Add config
        if config is not None:
            self.add_hparams(
                {k: v for k, v in vars(config).items() if isinstance(v, (int, float, str, bool, torch.Tensor))},
                {}
            )

    def log(self, data: Dict[str, Any], step: int) -> None:
        """Log each entry in data as its corresponding data type"""
        if self.enabled:
            for k, v in data.items():
                is_array = isinstance(v, np.ndarray) or isinstance(v, torch.Tensor)
                # Scalars
                if isinstance(v, Number) or (is_array and len(v.shape) == 0):
                    self.add_scalar(k, v, step)

                # Images
                elif is_array and len(v.shape) >= 3:
                    if len(v.shape) == 3:
                        self.add_image(k, v, step)
                    elif len(v.shape) == 4:
                        self.add_images(k, v, step)
                    else:
                        raise ValueError(f'Unsupported image shape: {v.shape}')

                else:
                    raise ValueError(f'Unsupported data type: {type(v)}')


