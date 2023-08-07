import functools
import os
import random
from datetime import datetime, timedelta
from argparse import Namespace
from numbers import Number
from typing import Any, Dict, Tuple

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
    checkpoint = {'model': model.state_dict(), 'config': config, 'step': step}
    torch.save(checkpoint, path)


def init_wandb(config, log_dir: str, group_name: str, job_type: str):
    dp_tag = "DP" if config.dp else "no_DP"
    wandb_tags = [config.model_type, config.dataset, config.protected_attr, dp_tag]
    os.makedirs(os.path.join(log_dir, "wandb"), exist_ok=True)
    run = wandb.init(project=config.wandb_project, config=config, group=group_name, dir=log_dir, tags=wandb_tags,
                     job_type=job_type, name="seed_" + str(config.seed),
                     mode="disabled" if (config.debug or config.disable_wandb) else "online")
    return run


def construct_log_dir(config, current_time):
    log_path = ""
    # job_type is always the same
    if config.protected_attr == "sex":
        jt = f"male_percent"
    else:
        jt = "old_percent"
    percent_as_string = str(config.protected_attr_percent).replace('.', '')
    if len(percent_as_string) > 3:
        percent_as_string = percent_as_string[:3]
    if percent_as_string.endswith("0"):
        percent_as_string = percent_as_string[:-1]
    jt += f"_{percent_as_string}"
    if config.job_type_mod:
        jt += f"-{config.job_type_mod}"
    if config.dp:
        jt += "_DP"
    # build log_dir and group_name
    log_path += f"{current_time}-{config.model_type}-{config.dataset}-{config.protected_attr}"
    if config.group_name_mod:
        log_path += f"-{config.group_name_mod}"
    if config.dp:
        log_path += "-DP"
    else:
        log_path += "-noDP"
    gn = log_path
    log_path = f"{log_path}/{jt}/seed_{config.seed}"
    log_path = f"logs/{log_path}"
    return log_path, gn, jt


class TensorboardLogger(SummaryWriter):
    def __init__(self, log_dir: str = None, config: Namespace = None, enabled: bool = True, comment: str = '',
                 purge_step: int = None, max_queue: int = 10, flush_secs: int = 120, filename_suffix: str = ''):
        self.enabled = enabled
        if self.enabled:
            super().__init__(log_dir=log_dir, comment=comment, purge_step=purge_step, max_queue=max_queue,
                             flush_secs=flush_secs, filename_suffix=filename_suffix)
        else:
            return

        # Add config
        if config is not None:
            self.add_hparams(
                {k: v for k, v in vars(config).items() if isinstance(v, (int, float, str, bool, torch.Tensor))}, {})

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


def log_time(remaining_time: float):
    time_left = int(remaining_time)
    time_duration = timedelta(seconds=time_left)
    days = time_duration.days
    hours = time_duration.seconds // 3600
    minutes = (time_duration.seconds // 60) % 60
    seconds = time_duration.seconds % 60
    return f"{days}d-{hours}h-{minutes}m-{seconds}s"


def get_subgroup_loss_weights(fraction: Tuple[float, float], mode="auroc", dp=False):
    # first in tuple is always "male_percent" or "old_percent"
    auroc_scores_dp = {"old": [0.77570003, 0.78220001, 0.79969999, 0.80450004, 0.8143],
                       "young": [0.65380001, 0.64079997, 0.62100002, 0.60250002, 0.5941], }
    auroc_scores_non_dp = {"old": [0.80762002, 0.83074002, 0.83918, 0.85316001, 0.8567],
                           "young": [0.77636, 0.76678, 0.74872, 0.73270002, 0.68015997], }
    scores = auroc_scores_dp if dp else auroc_scores_non_dp
    if mode == "auroc":
        auroc_at_frac = {(0.25, 0.75): (scores["old"][1], scores["young"][3]),
                         (0.50, 0.50): (scores["old"][2], scores["young"][2]),
                         (0.75, 0.25): (scores["old"][3], scores["young"][1]), }
        print(f"weight modifiers at fraction", fraction, "are",
              (1 / auroc_at_frac[fraction][0], 1 / auroc_at_frac[fraction][1]))
        return 1 / auroc_at_frac[fraction][0], 1 / auroc_at_frac[fraction][1]
    elif mode == "fraction":
        if fraction == (0.5, 0.5):
            print(f"weight modifiers at fraction", fraction, "are", (1, 2))
            return 1, 2
        else:
            print(f"weight modifiers at fraction", fraction, "are", (1 / fraction[0], 1 / fraction[1]))
            return 1 / fraction[0], 1 / fraction[1]
    elif mode == "fraction_od":
        # od = only_disadvantaged
        if fraction == (0.5, 0.5):
            print(f"weight modifiers at fraction", fraction, "are", (1, 2))
            return 1, 2
        else:
            print(f"weight modifiers at fraction", fraction, "are", (1, 1 / fraction[1]))
            return 1, 1 / fraction[1]
    elif mode == "fraction_rev":
        # pull the weights in the opposite direction
        print(f"weight modifiers at fraction", fraction, "are", (fraction[0], 1 / fraction[1]))
        return fraction[0], 1 / fraction[1]
    elif mode.startswith("old_down_weighted"):
        # get number at the end of the string
        weight = float(mode[-3:])
        print(f"weight modifiers at fraction", fraction, "are", (weight, 1))
        return weight, 1
